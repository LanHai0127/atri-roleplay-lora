"""LoRA / QLoRA 微调 → 亚托莉（8GB 显存可跑）。

用法：
    # 1.5B 基座（纯 bf16 LoRA，显存约 5GB）
    python scripts/train_lora.py --model models/Luminus-1.5B-Roleplay --max-steps 20

    # 3B 基座（自动 4bit QLoRA）
    python scripts/train_lora.py --model models/Qwen2.5-3B-Instruct --out outputs/atri-3b-lora

    # 在已有 LoRA 上继续强化（不重新初始化，适合第二轮 1 epoch）
    python scripts/train_lora.py --model models/Qwen2.5-3B-Instruct `
      --resume-adapter outputs/atri-v2-3b-lora --epochs 1 --lr 1e-4 `
      --out outputs/atri-v2-3b-lora-e2

关键取舍（RTX 5070 Laptop 8GB）：
- 1.5B：纯 bf16 LoRA，放得下，最快最稳。
- 3B ：bf16 权重就要 6.2GB，必须 4bit QLoRA（bnb 0.50 在 sm_120 上已验证可用）。
- gradient checkpointing + batch=1 + 梯度累积。
- 只对 assistant 段算 loss（system / user 全是 -100）。
- 手写 ChatML + 自定义 collator，不依赖 trl，避免版本漂移。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode_with_labels, load_tokenizer, project_root  # noqa: E402


def base_size_gb(model_dir: str) -> float:
    """按 safetensors 文件大小估算参数量对应的 bf16 体积（GB）。"""
    total = 0
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(model_dir, f))
    return total / 2**30



class SFTDataset(Dataset):
    def __init__(self, path: str, tok, max_len: int, skip_overlong: bool = False):
        self.items = []
        self.skipped = 0
        self.skipped_overlong = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids, labels, original_len = encode_with_labels(
                    tok, rec["messages"], max_len=max_len, return_original_len=True
                )
                if skip_overlong and original_len > max_len:
                    self.skipped_overlong += 1
                    continue
                if sum(1 for x in labels if x != -100) < 4:   # assistant 段太短，丢掉
                    self.skipped += 1
                    continue
                self.items.append({"input_ids": ids, "labels": labels})
        extra = f", 超长跳过 {self.skipped_overlong}" if skip_overlong else ""
        print(f"[data] {os.path.basename(path)}: {len(self.items)} 条可用, 跳过 {self.skipped}{extra}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n = maxlen - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * n)
            labels.append(x["labels"] + [-100] * n)
            attn.append([1] * len(x["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    root = project_root()
    ap.add_argument("--model", default=os.path.join(root, "models", "Qwen2.5-3B-Instruct"))
    ap.add_argument("--train", default=os.path.join(root, "data", "processed", "train.jsonl"))
    ap.add_argument("--valid", default=os.path.join(root, "data", "processed", "valid.jsonl"))
    ap.add_argument("--out", default=os.path.join(root, "outputs", "atri-3b-lora"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs，用于冒烟测试")
    ap.add_argument("--skip-overlong", action="store_true",
                    help="跳过超过 --max-len 的样本（不左截断），短样本训练用")
    ap.add_argument("--resume-adapter", default=None,
                    help="在已有 LoRA 适配器上继续训练（用于第二轮强化，不重新初始化 LoRA）")
    ap.add_argument("--load-4bit", choices=["auto", "on", "off"], default="auto",
                    help="4bit QLoRA：auto = 基座 > 4.5GB 时自动开")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.train):
        sys.exit(
            f"[!] 找不到训练数据：{args.train}\n"
            f"    先把语料放进 data/corpus/，再运行：\n"
            f"      python scripts/prepare_data.py\n"
            f"    只想先冒烟测试，可以指向示例数据：\n"
            f"      --train data/processed_demo/train.jsonl --valid data/processed_demo/valid.jsonl"
        )

    tokenizer = load_tokenizer(args.model)

    size_gb = base_size_gb(args.model)
    use_4bit = args.load_4bit == "on" or (args.load_4bit == "auto" and size_gb > 4.5)
    print(f"[model] 加载基座 {args.model}  ({size_gb:.2f} GB, 4bit={use_4bit}) ...")

    if use_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_cfg,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model.config.use_cache = False

    if args.resume_adapter:
        from peft import PeftModel
        print(f"[resume] 从已有适配器继续训练：{args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SFTDataset(args.train, tokenizer, args.max_len, skip_overlong=args.skip_overlong)
    valid_ds = (SFTDataset(args.valid, tokenizer, args.max_len, skip_overlong=args.skip_overlong)
                if os.path.exists(args.valid) else None)

    # transformers 5 去掉了 warmup_ratio，这里按实际步数算 5%
    steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.05))
    print(f"[plan] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        # paged_adamw_8bit 的 optimizer.pt 在 Windows 上会卡住（77MB adapter 能秒存，
        # optimizer.pt 却停在 0.59MB 且 CPU 空转）。冒烟/全量都只要 adapter，
        # 不保存 optimizer 状态，避免存档阶段假死。
        save_only_model=True,
        eval_strategy="epoch" if valid_ds else "no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[vram] 峰值显存 {peak:.2f} GB")
    print(f"\n[done] LoRA 已保存到 {args.out}")


if __name__ == "__main__":
    main()
