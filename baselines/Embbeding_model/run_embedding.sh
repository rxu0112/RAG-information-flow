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
DEVICE="${4:-cuda:0}"
PYTHON="${PYTHON:-${PYTHON_DEFAULT}}"

echo "======================================"
echo "Model:   ${MODEL_TAG}  (${MODEL_PATH})"
echo "Dataset: ${DATASET}  Device: ${DEVICE}"
echo "======================================"

"${PYTHON}" "${REPO_ROOT}/baselines/Embbeding_model/run_embedding_pipeline.py" \
  --model_tag "${MODEL_TAG}" \
  --dataset_tag "${DATASET}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}"
