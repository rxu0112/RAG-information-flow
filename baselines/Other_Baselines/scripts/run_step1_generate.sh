#!/usr/bin/env bash
set -euo pipefail

# ---- config (defaults) ----
MODEL_TAG_DEFAULT="Llama-3.2-3B-Instruct"                           # gemma-3-4B-it / Llama-3-8B-Instruct / Llama-3.2-3B-Instruct
MODEL_PATH_DEFAULT=PLACEHOLDER                            # e.g. /path/to/Llama-3.2-3B-Instruct
PYTHON_DEFAULT="/path/to/your/python"
REPO_ROOT="RAG-information-flow"
# ---------------------------

MODEL_TAG="${1:-${MODEL_TAG_DEFAULT}}"
MODEL_PATH="${2:-${MODEL_PATH_DEFAULT}}"
DATASET="${3:-squad2}"          # squad2 / hotpot / msmarco
SPLIT="${4:-val}"               # train / val / test
NUM_GENERATIONS="${5:-10}"
BATCH_SIZE="${6:-800}"
WORLD_SIZE="${7:-1}"
GPUS="${GPUS:-}"
PYTHON="${PYTHON:-${PYTHON_DEFAULT}}"

INPUT_FILE="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/${DATASET}_${SPLIT}_data.pt"
OUTPUT_DIR="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/baselines/temp/${SPLIT}/other"
OUTPUT_FILE="${OUTPUT_DIR}/answer_generations.jsonl"

mkdir -p "${OUTPUT_DIR}"
[ -n "${GPUS}" ] && export CUDA_VISIBLE_DEVICES="${GPUS}"
[ ! -f "${INPUT_FILE}" ] && echo "Not found: ${INPUT_FILE}" >&2 && exit 1

echo "======================================"
echo "Step 1: Generate multiple-round answers"
echo "======================================"
echo "Model:   ${MODEL_TAG}  (${MODEL_PATH})"
echo "Dataset: ${DATASET}  Split: ${SPLIT}"
echo "Input:   ${INPUT_FILE}"
echo "Output:  ${OUTPUT_FILE}"
echo "======================================"

"${PYTHON}" "${REPO_ROOT}/baselines/Other_Baselines/generate_new.py" \
  --model "${MODEL_PATH}" \
  --input_file "${INPUT_FILE}" \
  --result_fp "${OUTPUT_FILE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --world_size "${WORLD_SIZE}"
