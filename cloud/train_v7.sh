#!/usr/bin/env bash
# v7 云训练入口。和 cloud/train_7b.sh 的区别：走 scripts/train_v7.py（token 预算组批 +
# im_end 监督 + 零 datasets 依赖），参数默认值就是 v7 定稿那一组，改之前先读
# docs/v7_训练报告.md 第三节。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct}"
OUT="${OUT:-outputs/atri-v7-lora}"
TRAIN="${TRAIN:-data/train.jsonl}"
VALID="${VALID:-data/valid.jsonl}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
TOKEN_BUDGET="${TOKEN_BUDGET:-4500}"   # 24G 卡：4500≈19.4G，5000≈20.9G，6000 会 OOM
ACCUM="${ACCUM:-1}"
mkdir -p logs

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[!] 找不到基座：$MODEL_DIR"
  echo "    先运行 bash cloud/download_model.sh"
  exit 1
fi

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[train-v7] model=$MODEL_DIR out=$OUT epochs=$EPOCHS lr=$LR budget=$TOKEN_BUDGET"
python scripts/train_v7.py \
  --model "$MODEL_DIR" \
  --data "$TRAIN" \
  --val "$VALID" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --accum "$ACCUM" \
  --token-budget "$TOKEN_BUDGET" \
  --gc \
  2>&1 | tee "logs/cloud-train-v7-$(date +%Y%m%d-%H%M%S).log"
