#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct}"
OUT="${OUT:-outputs/atri-v5-7b-cloud}"
TRAIN="${TRAIN:-data/processed_v5/train.jsonl}"
VALID="${VALID:-data/processed_v5/valid.jsonl}"
MAX_LEN="${MAX_LEN:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
SEED="${SEED:-42}"
mkdir -p logs

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[!] 找不到基座：$MODEL_DIR"
  echo "    先运行 bash cloud/download_model.sh"
  exit 1
fi

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[train] model=$MODEL_DIR out=$OUT data=$TRAIN/$VALID max_len=$MAX_LEN epochs=$EPOCHS lr=$LR r=$LORA_R"
python scripts/train_lora.py \
  --model "$MODEL_DIR" \
  --train "$TRAIN" \
  --valid "$VALID" \
  --max-len "$MAX_LEN" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch 1 \
  --grad-accum 8 \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --load-4bit on \
  --seed "$SEED" \
  --out "$OUT" \
  2>&1 | tee "logs/cloud-train-$(date +%Y%m%d-%H%M%S).log"
