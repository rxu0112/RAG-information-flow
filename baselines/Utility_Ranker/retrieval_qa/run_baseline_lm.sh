#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

MODEL_SPEC="${1:-gemma}"
DATASET_TAG="${2:-hotpot}"
SPLIT_RAW="${3:-test}"
MODEL_NAME="${4:-/path/to/your/llm}"
TOP_N="${5:-5}"
OUTPUT_ROOT="${6:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${MODEL_SPEC}" in
  gemma|gemma-3-4B-it)
    MODEL_TAG="gemma-3-4B-it"
    ;;
  llama|llama-8b|Llama-3-8B-Instruct)
    MODEL_TAG="Llama-3-8B-Instruct"
    ;;
  3b|llama-3.2-3b|Llama-3.2-3B-Instruct)
    MODEL_TAG="Llama-3.2-3B-Instruct"
    ;;
  *)
    echo "Unsupported model: ${MODEL_SPEC}" >&2
    exit 1
    ;;
esac

case "${SPLIT_RAW}" in
  train) SPLIT="train" ;;
  val|valid|validation|dev) SPLIT="val" ;;
  test) SPLIT="test" ;;
  *)
    echo "Unsupported split: ${SPLIT_RAW}" >&2
    exit 1
    ;;
esac

if [ -z "${OUTPUT_ROOT}" ]; then
  OUTPUT_ROOT="${REPO_ROOT}/results/${MODEL_TAG}_${DATASET_TAG}/baselines/temp"
fi

RETRIEVAL_INPUT="${OUTPUT_ROOT}/${SPLIT}/utility_ranker/retrieval/absolute_retrieval.jsonl"
RESULT_FP="${OUTPUT_ROOT}/${SPLIT}/utility_ranker/run_baseline_lm.jsonl"

"${PYTHON_BIN}" "$ROOT_DIR/retrieval_qa/run_baseline_lm.py" \
   --model_name "${MODEL_NAME}" \
   --split "${SPLIT}" \
   --input_file "${RETRIEVAL_INPUT}" \
   --result_fp "${RESULT_FP}" \
   --prompt_name "chat_directRagQA_REAR3" \
   --chat_template \
   --top_n "${TOP_N}" \
   --temperature 0.0 \
   --top_p 1 \
   --max_new_tokens 50 \
   --do_stop \
   --logprobs 1 \
   --compute_pmi
