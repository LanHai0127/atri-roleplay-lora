"""环境自检：一条命令看清「能不能开训」。

    python scripts/check_env.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import project_root  # noqa: E402


def sec(t):
    print(f"\n=== {t} ===")


def main():
    root = project_root()
    print(f"项目根目录: {root}")
    print(f"Python    : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")

    sec("GPU")
    try:
        import torch
        print(f"torch     : {torch.__version__}  cuda={torch.version.cuda}")
        print(f"CUDA 可用 : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"显卡      : {torch.cuda.get_device_name(0)}")
            print(f"算力      : {torch.cuda.get_device_capability(0)}")
            print(f"显存      : {total/2**30:.1f} GB 总 / {free/2**30:.1f} GB 空闲")
            a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            _ = a @ a
            print("bf16 矩阵乘: OK")
    except Exception as e:
        print(f"[!] torch 异常: {e}")

    sec("关键依赖")
    for mod in ("transformers", "peft", "datasets", "accelerate", "safetensors", "numpy", "bitsandbytes"):
        try:
            m = __import__(mod)
            print(f"{mod:14s}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:14s}: [!] {e}")

    sec("基座模型（models/ 下）")
    mroot = os.path.join(root, "models")
    if not os.path.isdir(mroot):
        print("[!] 没有 models/ 目录")
    for name in sorted(os.listdir(mroot)):
        mp = os.path.join(mroot, name)
        if not os.path.isdir(mp):
            continue
        size = sum(os.path.getsize(os.path.join(mp, f))
                   for f in os.listdir(mp) if f.endswith(".safetensors"))
        vocab = "?"
        cfgp = os.path.join(mp, "config.json")
        if os.path.exists(cfgp):
            vocab = json.load(open(cfgp, encoding="utf-8")).get("vocab_size")
        ok = "OK" if size > 0 else "[!] 缺权重"
        mode = "4bit QLoRA（>4.5GB 自动）" if size / 2**30 > 4.5 else "bf16 LoRA"
        print(f"  {name:32s} {size/2**30:5.2f} GB  vocab={vocab}  {ok}  → {mode}")

    sec("数据")
    for d, pat in (("data/corpus", "你的语料"), ("data/processed", "编译产物")):
        p = os.path.join(root, d)
        files = []
        for base, _, fs in os.walk(p):
            files += [f for f in fs if f.endswith((".jsonl", ".json"))]
        print(f"{d:16s}: {len(files)} 个文件  ({pat})")

    sec("磁盘")
    free = shutil.disk_usage(root).free / 2**30
    print(f"剩余空间  : {free:.1f} GB  (至少留 8 GB 给 LoRA 输出/合并模型)")


if __name__ == "__main__":
    main()
