#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-models/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-outputs/atri-v5-7b-cloud}"
MERGED="${MERGED:-outputs/atri-v5-7b-merged}"
EXPORT_DIR="${EXPORT_DIR:-exports}"
LLAMA_DIR="${LLAMA_DIR:-third_party/llama.cpp}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggerganov/llama.cpp.git}"
SKIP_GGUF="${SKIP_GGUF:-0}"

if [ ! -d "$ADAPTER" ]; then
  echo "[!] 找不到 adapter：$ADAPTER"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

echo "[1/4] merge LoRA -> $MERGED"
python scripts/merge_lora.py \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --out "$MERGED"

if [ "$SKIP_GGUF" = "1" ]; then
  echo "[done] 只 merge，已跳过 GGUF"
  exit 0
fi

echo "[2/4] 准备 llama.cpp"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"
fi
python -m pip install -r "$LLAMA_DIR/requirements.txt"

echo "[3/4] 转 f16 GGUF"
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$EXPORT_DIR/atri-7b-f16.gguf" \
  --outtype f16

echo "[4/4] 量化 Q4_K_M"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-quantize
"$LLAMA_DIR/build/bin/llama-quantize" \
  "$EXPORT_DIR/atri-7b-f16.gguf" \
  "$EXPORT_DIR/atri-7b-Q4_K_M.gguf" \
  Q4_K_M

rm -f "$EXPORT_DIR/atri-7b-f16.gguf"
echo "[done] $EXPORT_DIR/atri-7b-Q4_K_M.gguf"
