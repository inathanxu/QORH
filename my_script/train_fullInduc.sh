#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash train.sh DATASET [GPU] [MAX_EPOCHS] [EARLY_STOPPING]

  DATASET         One of FB/NL/WK-{25,50,75,100}, e.g. FB-25
  GPU             CUDA device number, default 0
  MAX_EPOCHS      Maximum training epochs, default 200 (must be a positive integer)
  EARLY_STOPPING  Number of epochs to tolerate without improvement in the validation metric, default 10 (must be a positive integer)

Example: bash train.sh NL-50 0 200 10
EOF
}

if (( $# < 1 || $# > 4 )); then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KGS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${KGS_DIR}/.." && pwd -P)"

DATASET="${1^^}"
GPU="${2:-0}"
MAX_EPOCHS="${3:-200}"
EARLY_STOPPING="${4:-10}"

if [[ ! "${DATASET}" =~ ^(FB|NL|WK)-(25|50|75|100)$ ]]; then
    echo "Error: unsupported dataset label '${1}'." >&2
    usage
    exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU must be a non-negative integer, got '${GPU}'." >&2
    exit 2
fi
if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_EPOCHS must be a positive integer, got '${MAX_EPOCHS}'." >&2
    exit 2
fi
if [[ ! "${EARLY_STOPPING}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: EARLY_STOPPING must be a positive integer, got '${EARLY_STOPPING}'." >&2
    exit 2
fi

DATASET_DIR="${KGS_DIR}/data/${DATASET}_ind"
MODEL_DIR="${KGS_DIR}/saved_models"

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Error: dataset directory does not exist: ${DATASET_DIR}" >&2
    exit 1
fi

for split_file in train.txt msg.txt valid.txt test.txt; do
    if [[ ! -f "${DATASET_DIR}/${split_file}" ]]; then
        echo "Error: dataset is missing ${split_file}: ${DATASET_DIR}/${split_file}" >&2
        exit 1
    fi
done

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Error: could not find ${PROJECT_ROOT}/.venv/bin/python, python3, or python." >&2
    exit 1
fi

mkdir -p -- "${MODEL_DIR}"

echo "Dataset: ${DATASET_DIR}"
echo "Python: ${PYTHON}"
echo "GPU: ${GPU}; max epochs: ${MAX_EPOCHS}; early stopping: ${EARLY_STOPPING}"
echo "checkpoint: ${MODEL_DIR}/${DATASET}_ind.pt"

cd -- "${KGS_DIR}"
exec "${PYTHON}" "${KGS_DIR}/train.py" \
    --mode train \
    --datasets "${DATASET_DIR}" \
    --save_model \
    --model_dir "${MODEL_DIR}" \
    --gpu "${GPU}" \
    --max_epochs "${MAX_EPOCHS}" \
    --early_stopping "${EARLY_STOPPING}"
