#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash test_tradiInduc.sh DATASET [GPU] [NUM_LAYERS]

  DATASET     Base label of the paired directories, e.g. fb237_v1, WN18RR_v2, nell_v3
              Do not add the data/ path, _train_ind, or _ind suffix
  GPU         CUDA device number, default 0
  NUM_LAYERS  Number of relation encoder layers, default 3; must match the value used during training

Example: bash test_tradiInduc.sh fb237_v1 0 3
EOF
}

if (( $# < 1 || $# > 3 )); then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KGS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${KGS_DIR}/.." && pwd -P)"

DATASET="$1"
GPU="${2:-0}"
NUM_LAYERS="${3:-3}"

if [[ ! "${DATASET}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
   [[ "${DATASET}" == *_ind ]]; then
    echo "Error: DATASET must be a base label without a path or _ind suffix, got '${DATASET}'." >&2
    usage
    exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU must be a non-negative integer, got '${GPU}'." >&2
    exit 2
fi
if [[ ! "${NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_LAYERS must be a positive integer, got '${NUM_LAYERS}'." >&2
    exit 2
fi

TRAIN_DATASET_DIR="${KGS_DIR}/data/${DATASET}_train_ind"
TEST_DATASET_DIR="${KGS_DIR}/data/${DATASET}_ind"
MODEL_PATH="${KGS_DIR}/saved_models/${DATASET}_train_ind.pt"

for dataset_dir in "${TRAIN_DATASET_DIR}" "${TEST_DATASET_DIR}"; do
    if [[ ! -d "${dataset_dir}" ]]; then
        echo "Error: traditional inductive paired directory does not exist: ${dataset_dir}" >&2
        exit 1
    fi
    if [[ -f "${dataset_dir}/msg.txt" ]]; then
        echo "Error: ${dataset_dir} contains msg.txt; this script is only for traditional inductive data without msg.txt." >&2
        exit 1
    fi
    for split_file in train.txt valid.txt test.txt; do
        if [[ ! -f "${dataset_dir}/${split_file}" ]]; then
            echo "Error: dataset is missing ${split_file}: ${dataset_dir}/${split_file}" >&2
            exit 1
        fi
    done
done

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: training-side best checkpoint not found: ${MODEL_PATH}" >&2
    echo "Please run first: bash ${SCRIPT_DIR}/train_tradiInduc.sh ${DATASET} ${GPU} 200 10 ${NUM_LAYERS}" >&2
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

echo "Traditional inductive dataset: ${DATASET}"
echo "Training directory: ${TRAIN_DATASET_DIR}"
echo "Test directory: ${TEST_DATASET_DIR}"
echo "Python: ${PYTHON}"
echo "GPU: ${GPU}; relation encoder layers: ${NUM_LAYERS}"
echo "checkpoint: ${MODEL_PATH}"

cd -- "${KGS_DIR}"
exec "${PYTHON}" "${KGS_DIR}/train.py" \
    --mode zero-shot \
    --datasets "${TEST_DATASET_DIR}" \
    --model_path "${MODEL_PATH}" \
    --gpu "${GPU}" \
    --num_layers "${NUM_LAYERS}"
