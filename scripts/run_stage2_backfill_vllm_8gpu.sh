#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/GenSIvePFS/users/yfwang/miniconda3/envs/protllm-sft-vllm}"
MODEL_PATH="${MODEL_PATH:-/GenSIvePFS/users/model/gemma-4-31B-it}"
INPUT_PARQUET="${INPUT_PARQUET:-data/triples_formed.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/full_delivery_gemma4}"
SAMPLE_SIZE="${SAMPLE_SIZE:?SAMPLE_SIZE must be set}"
START_OFFSET="${START_OFFSET:-0}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
TP_SIZE="${TP_SIZE:-8}"
PARTITION_INDEX="${PARTITION_INDEX:?PARTITION_INDEX must be set}"
NUM_PARTITIONS="${NUM_PARTITIONS:?NUM_PARTITIONS must be set}"

set +u
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"
set -u

export PYTHONPATH=.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

python scripts/run_stage2_backfill_partition.py \
  --backend vllm \
  --sample-size "${SAMPLE_SIZE}" \
  --start-offset "${START_OFFSET}" \
  --input-parquet "${INPUT_PARQUET}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL_PATH}" \
  --stage2-batch-size "${STAGE2_BATCH_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --partition-index "${PARTITION_INDEX}" \
  --num-partitions "${NUM_PARTITIONS}" \
  "$@"
