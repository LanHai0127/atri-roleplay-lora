# -*- coding: utf-8 -*-
"""
v7 QLoRA 训练（4090 24G 优化版，零 datasets 依赖）。

为什么不用 datasets 库：这台机器的 pandas C 扩展是坏的
(`ImportError: C extension: pandas.util.version not built`)，datasets 靠 pandas 做
arrow <-> python 转换，导入即崩。语料本来就是纯 jsonl，自己读更快，也少一层拷贝。

针对 24G 的取舍：
1. **按 token 预算组批（不是按条数）**。语料长度 20~278 token 差 14 倍，固定 batch_size
   时短批只吃 6G、长批直接 24G 炸掉。改成 `批内样本数 × 批内最长 ≤ budget` 后，短样本
   一批能塞七八十条、长样本只塞十几条，显存峰值恒定（实测 19.4G / 24G）。
2. 长度分桶 —— 批内长度接近，padding 浪费从 ~10 倍压到 ~10%。
3. 默认关闭梯度检查点 —— 关掉后反向少一次前向重算，快约 30%；OOM 就加 --gc。
4. 逐轮 loss —— 只在 assistant 片段算 loss，多轮样本每一轮都算。
5. **assistant 正文后的 `<|im_end|>` 一起算 loss** —— 这是 v7 修掉的最致命 bug，漏了
   它模型只会「说什么」，不会「说到哪停」，loss 再低也会无限续写。
6. weight 降采样 —— 语料里给助手腔/知识问答打过 0.3~0.5 权重，按概率丢弃等价加权。

**学习率别按 batch_size 调**：改成 token 预算组批后每批条数从 16 变成 71，有效 batch
涨了 4 倍多，lr 还按小 batch 给（5e-5）就是严重欠拟合 —— 连训练集 11 字的答案都复现
不出来。v7 最终用的是 `--lr 2e-4 --accum 1`。

用法
----
  python scripts/train_v7.py --data data/train.jsonl --val data/valid.jsonl \
      --model models/Qwen2.5-7B-Instruct --out outputs/atri-v7-lora \
      --epochs 3 --accum 1 --gc --token-budget 4500 --lr 2e-4
"""
import argparse, json, math, os, random, time

import torch
from torch.utils.data import Dataset, Sampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback,
)

MAXLEN = 768


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_span(ids, sub):
    n = len(sub)
    if n == 0:
        return -1
    for s in range(len(ids) - n + 1):
        if ids[s:s + n] == sub:
            return s
    return -1


class ChatDS(Dataset):
    """纯内存数据集。rows 里每条是 {input_ids, labels, attention_mask, length}。"""

    column_names = ["input_ids", "labels", "attention_mask", "length"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class BucketSampler(Sampler):
    """按 token 预算组批：每批 sum(length) 不超过 budget，显存峰值恒定。

    按条数组批的问题是显存不可控 —— 语料长度 20~278 token 差 14 倍，固定 bs 时
    短批只吃 6G、长批直接吃 24G 炸掉。改成按 token 预算贪心装箱后，短样本一批能塞
    七八十条、长样本只塞十几条，显存恒定且 padding 浪费接近 0。
    """

    def __init__(self, lengths, budget=4000, max_batch=96, seed=0):
        self.lengths = lengths
        self.budget = budget
        self.max_batch = max_batch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def _pack(self, idx):
        """贪心装箱。成本 = 批内样本数 × 批内最长（padding 后的真实 token 量）。

        只累加 length 是不够的：96 条 52 token 的短样本，budget 看着才 4992，
        padding 到等长后实际要算 96×52 同样多，但 logits 和 CE 是按 batch×maxlen
        分配的。必须按 padding 后的量算，显存才可预测。
        """
        batches, cur, cur_max = [], [], 0
        for i in idx:
            L = self.lengths[i]
            if cur:
                n = len(cur) + 1
                m = max(cur_max, L)
                if n * m > self.budget or n > self.max_batch:
                    batches.append(cur)
                    cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, L)
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        perm = torch.randperm(len(idx), generator=g).tolist()
        idx = [idx[p] for p in perm]
        idx.sort(key=lambda i: self.lengths[i])       # 按长度排序 → 同批长度接近
        pieces = []
        for s in range(0, len(idx), 4000):            # 大段内打乱，避免长度单调
            part = idx[s:s + 4000]
            p2 = torch.randperm(len(part), generator=g).tolist()
            pieces.append([part[q] for q in p2])
        order = torch.randperm(len(pieces), generator=g).tolist()
        final = []
        for o in order:
            final.extend(pieces[o])
        batches = self._pack(final)
        bo = torch.randperm(len(batches), generator=g).tolist()
        for o in bo:
            yield batches[o]

    def __len__(self):
        return len(self._pack(sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])))


class BucketTrainer(Trainer):
    """把 DataLoader 换成按 token 预算组批的版本。"""

    def __init__(self, *a, token_budget=4000, max_batch=96, **kw):
        super().__init__(*a, **kw)
        self.token_budget = token_budget
        self.max_batch = max_batch
        self._sampler = None
        self._epoch_seen = -1

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        d = self.train_dataset
        if self._sampler is None:
            self._sampler = BucketSampler([r["length"] for r in d.rows],
                                          budget=self.token_budget,
                                          max_batch=self.max_batch)
        else:
            self._sampler.set_epoch(self._sampler.epoch + 1)
        return DataLoader(d, batch_sampler=self._sampler, collate_fn=self.data_collator,
                          num_workers=2, pin_memory=True)


class ProgressCb(TrainerCallback):
    """打印吞吐、显存、ETA —— 用来确认 GPU 真的在满负荷跑。"""

    def __init__(self, total_steps):
        self.total = total_steps
        self.t0 = None
        self.last = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = self.last = time.time()
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        now = time.time()
        dt = max(now - self.last, 1e-6)
        dn = state.global_step - self.last_step
        self.last, self.last_step = now, state.global_step
        sps = dn / dt
        eta = (self.total - state.global_step) / sps / 60 if sps > 0 else 0
        elapsed = (now - self.t0) / 60
        alloc = torch.cuda.max_memory_allocated() / 2**30
        log(f"step {state.global_step}/{self.total} loss={logs['loss']:.4f} "
            f"lr={logs.get('learning_rate', 0):.2e} | {sps:.2f} step/s | "
            f"显存峰值 {alloc:.1f}G | 已跑 {elapsed:.0f}min | ETA {eta:.0f}min")


def read_jsonl(path, tok, maxlen=MAXLEN, seed=0):
    """读语料 → 分词 → 逐轮打 label → 按 weight 降采样。"""
    rng = random.Random(seed)
    raw = [json.loads(l) for l in open(path, encoding="utf-8")]
    n0 = len(raw)
    t0 = time.time()

    # 关键：assistant 正文之后紧跟的 <|im_end|> 必须一起算 loss，
    # 否则模型只学会"说什么"，永远学不会"说到哪里停"——表现为回复无限延长。
    eos_ids = tok.encode("<|im_end|>", add_special_tokens=False) or [tok.eos_token_id]
    n_eos = 0

    rows, dropped_w, dropped_l = [], 0, 0
    for ex in raw:
        if rng.random() >= float(ex.get("weight", 1.0) or 1.0):
            dropped_w += 1
            continue
        msgs = ex["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(full, truncation=True, max_length=maxlen)["input_ids"]
        labels = [-100] * len(ids)
        cursor = 0
        for m in msgs:
            if m["role"] != "assistant":
                continue
            sub = tok(m["content"], add_special_tokens=False)["input_ids"]
            s = find_span(ids[cursor:], sub)
            if s < 0:
                continue
            s += cursor
            e = s + len(sub)
            labels[s:e] = ids[s:e]
            # 把紧随其后的结束符也纳入监督
            p = find_span(ids[e:], eos_ids)
            if 0 <= p <= 3:
                labels[e:e + p + len(eos_ids)] = ids[e:e + p + len(eos_ids)]
                cursor = e + p + len(eos_ids)
                n_eos += 1
            else:
                cursor = e
        if sum(1 for y in labels if y != -100) < 4:
            dropped_l += 1
            continue
        rows.append({"input_ids": ids, "labels": labels,
                     "attention_mask": [1] * len(ids), "length": len(ids)})

    ls = sorted(r["length"] for r in rows)
    log(f"[数据] {os.path.basename(path)} {n0} 条 → 保留 {len(rows)} "
        f"(weight 丢 {dropped_w} / 无有效 label 丢 {dropped_l}) "
        f"| 已训练结束符 {n_eos} 处 "
        f"| 长度 中位 {ls[len(ls)//2]} p95 {ls[int(.95*len(ls))]} max {ls[-1]} "
        f"| {sum(ls)/1e6:.2f}M tokens | 分词耗时 {time.time()-t0:.0f}s")
    return ChatDS(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--val", default="data/valid.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1,
                    help="默认 1：有效 batch 已由 token 预算决定，再叠 accum 会把 lr 稀释掉")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="token 预算下每批约 71 条，lr 要按大 batch 给；5e-5 会严重欠拟合")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=300)
    ap.add_argument("--log-steps", type=int, default=20)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8], help="量化位数")
    ap.add_argument("--gc", action="store_true", help="开启梯度检查点（显存紧张时用）")
    ap.add_argument("--token-budget", type=int, default=4500,
                    help="每批 token 预算，决定显存峰值：4500≈19.4G / 5000≈20.9G / 6000 OOM")
    ap.add_argument("--max-batch", type=int, default=96, help="单批样本数上限（短样本时防止过大）")
    ap.add_argument("--max-steps", type=int, default=-1, help="只跑这么多步就停，用于测速")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=(args.bits == 4), load_in_8bit=(args.bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gc)
    if args.gc:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    if args.gc:
        # use_reentrant=False 少一次 autograd 记录，比默认重入版快 10~15%
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    tr_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot_p = sum(p.numel() for p in model.parameters())
    log(f"[模型] 可训练 {tr_p/1e6:.1f}M / 总 {tot_p/1e9:.2f}B ({100*tr_p/tot_p:.2f}%) "
        f"| gc={'on' if args.gc else 'off'} | 权重显存 {torch.cuda.memory_allocated()/2**30:.2f}G")

    train = read_jsonl(args.data, tok, args.maxlen)
    valid = read_jsonl(args.val, tok, args.maxlen) if os.path.exists(args.val) else None

    # 步数由 token 预算决定：先估一遍批数
    probe_sampler = BucketSampler([r["length"] for r in train.rows],
                                  budget=args.token_budget, max_batch=args.max_batch)
    n_batches = len(probe_sampler)
    steps_per_epoch = math.ceil(n_batches / args.accum)
    total_steps = int(round(steps_per_epoch * args.epochs))
    avg_bs = len(train) / max(n_batches, 1)
    log(f"[计划] token预算={args.token_budget} → 约 {n_batches} 批/epoch "
        f"(平均 {avg_bs:.0f} 条/批) → {steps_per_epoch} 步/epoch × {args.epochs} "
        f"= {total_steps} 步")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=args.log_steps, save_steps=args.save_steps, save_total_limit=2,
        bf16=True, gradient_checkpointing=args.gc, optim="paged_adamw_8bit",
        eval_strategy="no", report_to=[], remove_unused_columns=False,
        dataloader_num_workers=2, dataloader_pin_memory=True,
        save_only_model=True, max_grad_norm=1.0,
    )
    if args.max_steps > 0:
        targs.max_steps = args.max_steps
        targs.save_strategy = "no"
    trainer = BucketTrainer(
        model=model, args=targs, train_dataset=train, eval_dataset=valid,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        callbacks=[ProgressCb(total_steps)],
        token_budget=args.token_budget, max_batch=args.max_batch,
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    dur = (time.time() - t0) / 60
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"[完成] adapter → {args.out} | 总耗时 {dur:.1f} min | 显存峰值 {peak:.2f}G")

    meta = {"minutes": round(dur, 1), "steps": total_steps, "bs": args.bs,
            "accum": args.accum, "epochs": args.epochs, "samples": len(train),
            "peak_gb": round(peak, 2), "gc": args.gc, "lr": args.lr, "r": args.r,
            "maxlen": args.maxlen, "finished": time.strftime("%F %T")}
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log("[meta] " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
