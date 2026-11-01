"""亚托莉 v7 命令行聊天 —— 无 system prompt、短回复。

v7 和 v6 的推理路径**不一样**，混用会退化，原因如下：

1. **不要 system prompt**。v7 的 21,284 条语料 100% 不带 system，人格是训进权重的。
   加一条 system 会把模型推回「助手模式」，这是 v6 时代 issue #1 的根因。
2. **user 内容前面要加「花海：」**。训练时每条 user 都是这个格式，推理端不补前缀
   等于训练/推理分布错位，回复会变长、变客套。
3. **max_new_tokens 给小**。v7 回复中位 11 字，给 180 会让它有机会越写越长。

用法：
    # 交互
    python scripts/chat_v7.py --base models/Qwen2.5-7B-Instruct \
        --adapter models/atri-v7-lora --load-4bit on

    # 单句
    python scripts/chat_v7.py --base models/Qwen2.5-7B-Instruct \
        --adapter models/atri-v7-lora --once "我好累"

交互内命令：/reset 清空上下文，/temp 0.8 调温度，/quit 退出
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

IM_END_ID = 151645
EOT_ID = 151643
USER_PREFIX = "花海："


def base_size_gb(d: str) -> float:
    t = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                t += os.path.getsize(os.path.join(d, f))
    return t / 2**30


def load_model(base: str, adapter: str | None, load_4bit: str = "auto"):
    size = base_size_gb(base)
    use_4bit = load_4bit == "on" or (load_4bit == "auto" and size > 4.5)
    print(f"[加载] {base} ({size:.1f} GB, 4bit={use_4bit})"
          + (f" + LoRA {adapter}" if adapter else ""), file=sys.stderr)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    return model


def add_prefix(text: str) -> str:
    """补「花海：」前缀。已经是这个开头就不重复加。"""
    t = text.strip()
    if t.startswith(USER_PREFIX):
        return t
    return USER_PREFIX + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基座模型目录")
    ap.add_argument("--adapter", default=None, help="v7 LoRA 目录")
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="v7 回复中位 11 字，给太大容易越写越长")
    ap.add_argument("--once", default=None, help="只回答一句就退出")
    ap.add_argument("--load-4bit", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.base, args.adapter, args.load_4bit)
    history: list[dict] = []

    def build():
        # 注意：这里故意不放 system。见文件头第 1 条。
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").to("cuda")

    def reply(user_text: str, stream: bool = False) -> str:
        history.append({"role": "user", "content": add_prefix(user_text)})
        inputs = build()
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temp > 0,
            temperature=max(args.temp, 1e-5),
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            eos_token_id=[IM_END_ID, EOT_ID],
            pad_token_id=tok.pad_token_id,
        )
        if not stream:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
        else:
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            th = Thread(target=model.generate, kwargs={**inputs, **gen_kwargs, "streamer": streamer})
            th.start()
            chunks = []
            for piece in streamer:
                chunks.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            th.join()
            text = "".join(chunks).strip()
            print()
        # 只取第一行：模型偶尔会在回复后补一段，多轮时会污染上下文
        text = text.split("\n")[0].strip() or "……"
        history.append({"role": "assistant", "content": text})
        return text

    if args.once:
        print(reply(args.once))
        return

    print("=== 亚托莉 v7 ===")
    print("输入 /quit 退出，/reset 清空上下文，/temp 0.8 调温度\n")
    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/reset":
            history.clear()
            print("（上下文已清空）")
            continue
        if line.startswith("/temp"):
            try:
                args.temp = float(line[5:].strip())
                print(f"（温度 = {args.temp}）")
            except ValueError:
                print("用法：/temp 0.8")
            continue
        print("亚托莉 > ", end="", flush=True)
        print(reply(line))


if __name__ == "__main__":
    main()
