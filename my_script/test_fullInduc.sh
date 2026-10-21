#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash test.sh DATASET [GPU]

  DATASET  One of FB/NL/WK-{25,50,75,100}, e.g. FB-25
  GPU      CUDA device number, default 0

Example: bash test.sh NL-50 0
EOF
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KGS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${KGS_DIR}/.." && pwd -P)"

DATASET="${1^^}"
GPU="${2:-0}"

if [[ ! "${DATASET}" =~ ^(FB|NL|WK)-(25|50|75|100)$ ]]; then
    echo "Error: unsupported dataset label '${1}'." >&2
    usage
    exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU must be a non-negative integer, got '${GPU}'." >&2
    exit 2
fi

DATASET_DIR="${KGS_DIR}/data/${DATASET}_ind"
MODEL_PATH="${KGS_DIR}/saved_models/${DATASET}_ind.pt"

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

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: checkpoint not found: ${MODEL_PATH}" >&2
    echo "Please run first: bash ${SCRIPT_DIR}/train.sh ${DATASET} ${GPU}" >&2
    exit 1
fi

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

echo "Dataset: ${DATASET_DIR}"
echo "Python: ${PYTHON}"
echo "GPU: ${GPU}"
echo "checkpoint: ${MODEL_PATH}"

cd -- "${KGS_DIR}"
exec "${PYTHON}" "${KGS_DIR}/train.py" \
    --mode zero-shot \
    --datasets "${DATASET_DIR}" \
    --gpu "${GPU}" \
    --model_path "${MODEL_PATH}"
