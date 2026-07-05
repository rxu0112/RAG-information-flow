#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

MODEL_SPEC="${1:-gemma}"
DATASET_TAG="${2:-hotpot}"
SPLIT_RAW="${3:-test}"
TOP_N="${4:-5}"
OUTPUT_ROOT="${5:-}"
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

SPLIT_DIR="${OUTPUT_ROOT}/${SPLIT}/utility_ranker"
CONTEXT_FILE="${SPLIT_DIR}/context.jsonl"
ABSOLUTE_FILE="${SPLIT_DIR}/absolute.jsonl"
EMBED_DIR="${SPLIT_DIR}/embeddings"
RETRIEVAL_DIR="${SPLIT_DIR}/retrieval"

"${PYTHON_BIN}" "$ROOT_DIR/contriever/passage_retrieval.py" \
    --model_name_or_path "$ROOT_DIR/contriever_msmarco" \
    --passages "$CONTEXT_FILE" \
    --passages_embeddings "$EMBED_DIR/passages_*" \
    --data "$ABSOLUTE_FILE" \
    --output_dir "$RETRIEVAL_DIR" \
    --n_docs "$TOP_N"
