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
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-${PYTHON_DEFAULT}}"

INPUT_DIR="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/baselines/temp/${SPLIT}/other"
GENERATION_FILE="${INPUT_DIR}/answer_generations.jsonl"
ALL_DATA="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET}/${DATASET}_${SPLIT}_data.pt"
OUTPUT_FILE="${INPUT_DIR}/uncertainty_scores.json"

export CUDA_VISIBLE_DEVICES="${GPUS}"
[ ! -f "${GENERATION_FILE}" ] && echo "Not found: ${GENERATION_FILE}" >&2 && exit 1

echo "======================================"
echo "Step 2: Compute uncertainty metrics"
echo "======================================"
echo "Model:   ${MODEL_TAG}  (${MODEL_PATH})"
echo "Dataset: ${DATASET}  Split: ${SPLIT}  GPUs: ${GPUS}"
echo "Input:   ${GENERATION_FILE}"
echo "Output:  ${OUTPUT_FILE}"
echo "======================================"

"${PYTHON}" "${REPO_ROOT}/baselines/Other_Baselines/compute_uncertainty_measures_flexible.py" \
  --dataset "${GENERATION_FILE}" \
  --all_data "${ALL_DATA}" \
  --output "${OUTPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --gpus "${GPUS}" \
  --compute_all
