#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash train_tradiInduc.sh DATASET [GPU] [MAX_EPOCHS] [EARLY_STOPPING] [NUM_LAYERS]

  DATASET         Base label of the paired directories, e.g. fb237_v1, WN18RR_v2, nell_v3
                  Do not add the data/ path, _train_ind, or _ind suffix
  GPU             CUDA device number, default 0
  MAX_EPOCHS      Maximum training epochs, default 200 (must be a positive integer)
  EARLY_STOPPING  Number of epochs to tolerate without improvement in the validation metric, default 10 (must be a positive integer)
  NUM_LAYERS      Number of relation encoder layers, default 3; must use the same value when testing

Example: bash train_tradiInduc.sh fb237_v1 0 200 10 3
EOF
}

if (( $# < 1 || $# > 5 )); then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KGS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${KGS_DIR}/.." && pwd -P)"

DATASET="$1"
GPU="${2:-0}"
MAX_EPOCHS="${3:-200}"
EARLY_STOPPING="${4:-10}"
NUM_LAYERS="${5:-3}"

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
if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_EPOCHS must be a positive integer, got '${MAX_EPOCHS}'." >&2
    exit 2
fi
if [[ ! "${EARLY_STOPPING}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: EARLY_STOPPING must be a positive integer, got '${EARLY_STOPPING}'." >&2
    exit 2
fi
if [[ ! "${NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_LAYERS must be a positive integer, got '${NUM_LAYERS}'." >&2
    exit 2
fi

TRAIN_DATASET_DIR="${KGS_DIR}/data/${DATASET}_train_ind"
TEST_DATASET_DIR="${KGS_DIR}/data/${DATASET}_ind"
MODEL_DIR="${KGS_DIR}/saved_models"
MODEL_PATH="${MODEL_DIR}/${DATASET}_train_ind.pt"

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

echo "Traditional inductive dataset: ${DATASET}"
echo "Training directory: ${TRAIN_DATASET_DIR}"
echo "Paired test directory: ${TEST_DATASET_DIR}"
echo "Python: ${PYTHON}"
echo "GPU: ${GPU}; max epochs: ${MAX_EPOCHS}; early stopping: ${EARLY_STOPPING}"
echo "Relation encoder layers: ${NUM_LAYERS}"
echo "checkpoint: ${MODEL_PATH}"

cd -- "${KGS_DIR}"
exec "${PYTHON}" "${KGS_DIR}/train.py" \
    --mode train \
    --datasets "${TRAIN_DATASET_DIR}" \
    --save_model \
    --model_dir "${MODEL_DIR}" \
    --gpu "${GPU}" \
    --max_epochs "${MAX_EPOCHS}" \
    --early_stopping "${EARLY_STOPPING}" \
    --num_layers "${NUM_LAYERS}"
