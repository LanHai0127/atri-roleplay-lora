#!/usr/bin/env bash
# 启动 atri v7 的 llama-server（Q8 底模 + 运行时挂载 LoRA，不合并权重）。
#
# 需要的环境变量（不设就用下面的默认值）：
#   ATRI_BIN      llama.cpp 编译产物目录
#   ATRI_BASE     Q8_0 底模 gguf（纯 Qwen2.5-7B-Instruct 转的，没合并过 LoRA）
#   ATRI_LORA     v7 LoRA 的 gguf（由 convert_lora_to_gguf.py --outtype f16 产出）
#   ATRI_DIR      模板所在目录
set -e

BIN="${ATRI_BIN:-$HOME/atri-deploy/build/llama.cpp/build/bin}"
DIR="${ATRI_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BASE="${ATRI_BASE:-$DIR/Qwen2.5-7B-Instruct-Q8_0.gguf}"
LORA="${ATRI_LORA:-$DIR/atri-v7-lora.gguf}"
THREADS="${ATRI_THREADS:-$(nproc)}"
CTX="${ATRI_CTX:-16384}"

exec "$BIN/llama-server" \
  -m "$BASE" \
  --lora "$LORA" \
  --host 127.0.0.1 \
  --port 8088 \
  -t "$THREADS" \
  -c "$CTX" \
  --parallel 1 \
  -fa auto \
  --jinja \
  --chat-template-file "$DIR/chat-template.jinja" \
  --alias atri
