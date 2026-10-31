#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Python / CUDA"
python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY

echo "[2/3] pip install"
python -m pip install --upgrade pip
python -m pip install -r cloud/requirements.txt

echo "[3/3] import check"
python - <<'PY'
import transformers, peft, bitsandbytes, accelerate, datasets
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
PY

echo "[done] setup ok"
