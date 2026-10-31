"""亚托莉 命令行聊天（支持流式输出、多轮上下文）。

三种用法：
    # 1) 基座 + LoRA（训练完立刻试，不用合并）
    python scripts/chat.py --adapter outputs/atri-lora-v1

    # 2) 合并后的完整模型
    python scripts/chat.py --model outputs/atri-merged-v1

    # 3) 单句测试（脚本/自动化用）
    python scripts/chat.py --adapter outputs/atri-lora-v1 --once "亚托莉，你在做什么？"

交互内命令：
    /reset        清空上下文
    /scene <文本> 设置场景（例如 /scene 傍晚的海边）
    /role <名字>  设置对面是谁（默认 花海）
    /temp 0.8     调整温度
    /quit         退出
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_system_prompt, clean_reply, ensure_identity, ensure_name_address,  # noqa: E402
                    identity_reply, knowledge_reply, load_tokenizer, messages_to_text, project_root,
                    role_reply, safety_reply)

IM_END_ID = 151645
EOT_ID = 151643


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
    print(f"[加载] {base} ({size:.1f} GB, 4bit={use_4bit})" + (f" + LoRA {adapter}" if adapter else ""),
          file=sys.stderr)
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


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="合并后的完整模型目录")
    ap.add_argument("--base", default=os.path.join(root, "models", "Qwen2.5-3B-Instruct"))
    ap.add_argument("--adapter", default=None, help="LoRA 目录")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--role", default="花海")
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--ngram", type=int, default=3,
                    help="禁止重复的 n-gram 长度（0=关闭；小模型建议 3，防复读）")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--once", default=None, help="只回答一句就退出")
    ap.add_argument("--load-4bit", choices=["auto", "on", "off"], default="auto",
                    help="4bit 推理：auto = 模型 > 4.5GB 时自动开（8GB 显存跑 3B 用）")
    args = ap.parse_args()

    base = args.model or args.base
    tok = load_tokenizer(base)
    model = load_model(base, None if args.model else args.adapter, args.load_4bit)

    scene, role = args.scene, args.role
    history: list[dict] = []

    def build():
        msgs = [{"role": "system", "content": build_system_prompt(scene, role)}] + history
        return tok(messages_to_text(tok, msgs, add_generation_prompt=True), return_tensors="pt").to("cuda")

    def reply(user_text: str, stream: bool = False) -> str:
        history.append({"role": "user", "content": user_text})
        guard = (safety_reply(user_text) or identity_reply(user_text)
                 or knowledge_reply(user_text) or role_reply(user_text))
        if guard:
            print(f"亚托莉：{guard}\n")
            history.append({"role": "assistant", "content": guard})
            return guard
        inputs = build()
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temp > 0,
            temperature=max(args.temp, 1e-5),
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            no_repeat_ngram_size=args.ngram,
            eos_token_id=[IM_END_ID, EOT_ID],
            pad_token_id=tok.pad_token_id,
        )
        if not stream:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
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
        text = clean_reply(text)
        text = ensure_name_address(user_text, text)
        text = ensure_identity(user_text, text)
        history.append({"role": "assistant", "content": text})
        return text

    if args.once:
        print(reply(args.once, stream=False))
        return

    print("=== 亚托莉 ===")
    print("输入 /quit 退出，/reset 清空，/scene <场景> 设置场景，/role <名字> 设置对方\n")
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
        if line.startswith("/scene"):
            scene = line[6:].strip() or None
            print(f"（场景 = {scene}）")
            continue
        if line.startswith("/role"):
            role = line[5:].strip() or None
            print(f"（对面 = {role}）")
            continue
        if line.startswith("/temp"):
            try:
                args.temp = float(line[5:].strip())
                print(f"（温度 = {args.temp}）")
            except ValueError:
                print("用法：/temp 0.8")
            continue
        print("亚托莉 > ", end="")
        print(reply(line, stream=False))


if __name__ == "__main__":
    main()
