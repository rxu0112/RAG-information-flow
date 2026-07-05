#!/usr/bin/env bash
set -euo pipefail

# ---- config (defaults) ----
MODEL_TAG_DEFAULT="Llama-3.2-3B-Instruct"                           # gemma-3-4B-it / Llama-3-8B-Instruct / Llama-3.2-3B-Instruct
MODEL_PATH_DEFAULT=MODEL_PATH_PLACEHOLDER                            # e.g. /path/to/Llama-3.2-3B-Instruct
PYTHON_DEFAULT="/path/to/your/python"
REPO_ROOT="RAG-information-flow"
# ---------------------------

MODEL_TAG="${1:-${MODEL_TAG_DEFAULT}}"
MODEL_PATH="${2:-${MODEL_PATH_DEFAULT}}"
DATASET="${3:-squad2}"          # squad2 / hotpot / msmarco
SPLIT="${4:-val}"               # train / val / test
GPU_ID="${5:-0}"
PYTHON="${PYTHON:-${PYTHON_DEFAULT}}"

INPUT_FILE="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/${DATASET}_${SPLIT}_data.pt"
OUTPUT_DIR="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/baselines/temp/${SPLIT}/focus"

mkdir -p "${OUTPUT_DIR}"
[ ! -f "${INPUT_FILE}" ] && echo "Not found: ${INPUT_FILE}" >&2 && exit 1

echo "======================================"
echo "Model:   ${MODEL_TAG}  (${MODEL_PATH})"
echo "Dataset: ${DATASET}  Split: ${SPLIT}  GPU: ${GPU_ID}"
echo "Input:   ${INPUT_FILE}"
echo "Output:  ${OUTPUT_DIR}"
echo "======================================"

"${PYTHON}" "${REPO_ROOT}/baselines/Focus/Focus.py" \
  --model_tag "${MODEL_TAG}" \
  --model_path "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --gpu_id "${GPU_ID}"
