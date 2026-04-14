#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-/GenSIvePFS/users/yfwang/miniconda3/envs/protllm-sft-vllm}"
ENV_READY_MARKER_REL=".codex_env_ready_v3"
LOCK_FILE="/GenSIvePFS/users/yfwang/.tmp/protllm-sft-vllm-setup.lock"

set +u
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y -p "${ENV_PREFIX}" python=3.12 pip
fi

conda activate "${ENV_PREFIX}"
set -u
export TMPDIR="${TMPDIR:-/GenSIvePFS/users/yfwang/.tmp/pip-tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/GenSIvePFS/users/yfwang/.cache/pip}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "$(dirname "${LOCK_FILE}")"

READY_MARKER="${ENV_PREFIX}/${ENV_READY_MARKER_REL}"
if [[ -f "${READY_MARKER}" ]]; then
  echo "Environment already ready: ${ENV_PREFIX}"
  exit 0
fi

exec 9>"${LOCK_FILE}"
flock 9

if [[ -f "${READY_MARKER}" ]]; then
  echo "Environment already ready after waiting for lock: ${ENV_PREFIX}"
  exit 0
fi

python -m pip install --upgrade pip

# Install general dependencies first, excluding the model-runtime stack that we pin below.
grep -Ev '^(vllm|transformers)>=' requirements.txt > "${TMPDIR}/requirements.base.txt"
python -m pip install --no-cache-dir -r "${TMPDIR}/requirements.base.txt"

# Gemma 4 in this workspace needs a bleeding-edge Transformers build to recognize
# the architecture. We still serialize installs so concurrent task startup does
# not corrupt the shared environment.
python -m pip install --no-cache-dir -U --pre \
  "vllm" \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129
python -m pip install --no-cache-dir "transformers==5.5.0"

touch "${READY_MARKER}"

echo "Environment ready: ${ENV_PREFIX}"
