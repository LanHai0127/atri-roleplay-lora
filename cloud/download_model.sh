#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p "$(dirname "$MODEL_DIR")"

if [ -f "$MODEL_DIR/config.json" ]; then
  echo "[skip] 已存在 $MODEL_DIR"
  exit 0
fi

echo "[download] $MODEL_ID -> $MODEL_DIR"

if command -v modelscope >/dev/null 2>&1; then
  echo "[source] ModelScope"
  modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
else
  echo "[source] hf-mirror（可设 HF_ENDPOINT 覆盖）"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

python - <<PY
import os, json
p = os.path.join("$MODEL_DIR", "config.json")
assert os.path.exists(p), p
cfg = json.load(open(p, encoding="utf-8"))
print("[ok]", cfg.get("model_type"), cfg.get("num_hidden_layers"), "layers", cfg.get("hidden_size"), "hidden")
PY
