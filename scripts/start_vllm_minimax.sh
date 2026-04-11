#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/data/cloudroot/minimax-m2.5-deploy/models/MiniMax-M2.5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
CONTEXT_LEN="${MAX_MODEL_LEN:-32768}"

export SAFETENSORS_FAST_GPU=1

vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${CONTEXT_LEN}" \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
