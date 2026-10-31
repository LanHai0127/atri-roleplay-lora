"""把 LoRA 权重合并回基座，导出可直接推理的完整模型。

用法：
    python scripts/merge_lora.py --adapter outputs/atri-lora-v1 --out outputs/atri-merged-v1
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_tokenizer, project_root  # noqa: E402


def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(root, "models", "Qwen2.5-3B-Instruct"))
    ap.add_argument("--adapter", default=os.path.join(root, "outputs", "atri-3b-lora"))
    ap.add_argument("--out", default=os.path.join(root, "outputs", "atri-3b-merged"))
    args = ap.parse_args()

    print("[1/3] 加载基座 (CPU) ...")
    # 7B bf16 约 14.2 GB；low_cpu_mem_usage 避免加载时再复制一份完整 state_dict。
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    tokenizer = load_tokenizer(args.base)

    print("[2/3] 挂载并合并 LoRA ...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
    # PEFT 0.21.x 的 PeftModel 包装层不直接暴露 merge_and_unload；
    # 真正的方法是 LoraModel（model.base_model）上的。
    merged = None
    for obj in (model, getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None)):
        if obj is not None and hasattr(obj, "merge_and_unload"):
            merged = obj.merge_and_unload()
            break
    if merged is None:
        raise RuntimeError("找不到 merge_and_unload；请检查 PEFT 版本")
    model = merged

    print("[3/3] 保存 ...")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
