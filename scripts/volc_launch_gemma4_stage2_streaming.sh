#!/usr/bin/env bash
set -euo pipefail

cd /GenSIvePFS/users/yfwang/protllm-sft-data-synthesis

echo "SFT_STAGE2_STREAM_JOB_STARTED"
echo "PWD=$(pwd)"
echo "MODEL_PATH=${MODEL_PATH:-/GenSIvePFS/users/model/gemma-4-31B-it}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE:-256}"
echo "START_OFFSET=${START_OFFSET:-0}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/full_delivery_gemma4}"

bash scripts/setup_env.sh "${ENV_PREFIX:-/GenSIvePFS/users/yfwang/miniconda3/envs/protllm-sft-vllm}"
bash scripts/run_stage2_streaming_vllm_8gpu.sh
