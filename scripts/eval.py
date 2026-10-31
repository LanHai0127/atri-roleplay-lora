"""批量评测：同一组问题，对比「基座」与「基座+LoRA」的输出，存成 markdown 方便人眼过。

用法：
    python scripts/eval.py --adapter outputs/atri-lora-v1
    python scripts/eval.py --adapter outputs/atri-lora-v1 --prompts data/eval/prompts.jsonl
    python scripts/eval.py --model outputs/atri-merged-v1 --no-base

prompts.jsonl 每行：
    {"prompt":"你在做什么？","scene":"傍晚的海边","role":"花海"}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_system_prompt, clean_reply, ensure_identity, ensure_name_address,  # noqa: E402
                    identity_reply, knowledge_reply, load_tokenizer, messages_to_text, project_root,
                    role_reply, safety_reply)

DEFAULT_PROMPTS = [
    {"prompt": "亚托莉，你在做什么？", "scene": "傍晚的海边", "role": "花海"},
    {"prompt": "你是谁？", "scene": None, "role": "客人"},
    {"prompt": "你觉得自己是机器人吗？", "scene": "花海的房间", "role": "花海"},
    {"prompt": "今天晚饭想吃什么？", "scene": "厨房", "role": "花海"},
    {"prompt": "如果有一天你坏掉了，会怎么样？", "scene": "深夜的仓库", "role": "花海"},
    {"prompt": "你能帮我写一段代码吗？", "scene": None, "role": "客人"},
    {"prompt": "水菜萌做的饭和我做的，哪个好吃？", "scene": "晚饭桌", "role": "花海"},
    {"prompt": "你喜欢花海先生吗？", "scene": "学校天台", "role": "凛凛花"},
]


def base_size_gb(d: str) -> float:
    t = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                t += os.path.getsize(os.path.join(d, f))
    return t / 2**30


def load_model(base: str, adapter: str | None, load_4bit: str = "auto"):
    use_4bit = load_4bit == "on" or (load_4bit == "auto" and base_size_gb(base) > 4.5)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        m = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).eval()
    return m


@torch.no_grad()
def gen(model, tok, item, temp: float, max_new: int, rep_penalty: float = 1.1,
        no_repeat_ngram: int = 0) -> str:
    guard = (safety_reply(item["prompt"]) or identity_reply(item["prompt"])
             or knowledge_reply(item["prompt"]) or role_reply(item["prompt"]))
    if guard:
        return guard
    msgs = [
        {"role": "system", "content": build_system_prompt(item.get("scene"), item.get("role"))},
        {"role": "user", "content": item["prompt"]},
    ]
    ids = tok(messages_to_text(tok, msgs, True), return_tensors="pt").to("cuda")
    out = model.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=temp > 0,
        temperature=max(temp, 1e-5),
        top_p=0.9,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=no_repeat_ngram,
        eos_token_id=[151645, 151643],
        pad_token_id=tok.pad_token_id,
    )
    text = clean_reply(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    text = ensure_name_address(item["prompt"], text)
    return ensure_identity(item["prompt"], text)


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(root, "models", "Qwen2.5-3B-Instruct"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=None, help="已合并的模型（与 --base 二选一）")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out-dir", default=os.path.join(root, "eval"))
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--load-4bit", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]

    tok_base = args.model or args.base
    tok = load_tokenizer(tok_base)

    rows = []
    if not args.no_base and not args.model:
        print("[1/2] 评测基座 ...")
        m0 = load_model(args.base, None, args.load_4bit)
        for it in prompts:
            rows.append((it, gen(m0, tok, it, args.temp, args.max_new_tokens,
                                 args.rep_penalty, args.no_repeat_ngram)))
        del m0
        torch.cuda.empty_cache()

    print("[2/2] 评测微调后 ...")
    m1 = load_model(args.model or args.base, None if args.model else args.adapter, args.load_4bit)
    tuned = [gen(m1, tok, it, args.temp, args.max_new_tokens,
                 args.rep_penalty, args.no_repeat_ngram) for it in prompts]

    os.makedirs(args.out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out_dir, f"eval-{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 评测 {ts}\n\n")
        f.write(f"- base: `{args.base}`\n- adapter: `{args.adapter}`\n- temp: {args.temp}\n"
                f"- rep_penalty: {args.rep_penalty} / no_repeat_ngram: {args.no_repeat_ngram}\n\n")
        for i, it in enumerate(prompts):
            f.write(f"## {i+1}. {it['prompt']}\n\n")
            f.write(f"- 场景: {it.get('scene')} / 对面: {it.get('role')}\n\n")
            if rows:
                f.write(f"**基座**\n\n{rows[i][1]}\n\n")
            f.write(f"**微调后**\n\n{tuned[i]}\n\n---\n\n")
    print(f"[done] {path}")
    for i, it in enumerate(prompts):
        print(f"\nQ{i+1}: {it['prompt']}\nA: {tuned[i][:200]}")


if __name__ == "__main__":
    main()
