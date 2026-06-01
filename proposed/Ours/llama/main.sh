#!/usr/bin/env bash
set -uo pipefail   
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET
export TORCH_NCCL_DESYNC_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_TAG=${MODEL_TAG:-Llama-3.2-3B-Instruct} # choose from Llama-3.2-3B-Instruct, Llama-3-8B-Instruct 
DATASET_TAG=${DATASET_TAG:-hotpot} # choose from squad2, hotpot, msmarco
MODEL_PATH=${MODEL_PATH:-MODEL_PATH_PLACEHOLDER} # path of args.model
I_BLOCK=${I_BLOCK:-100}    # Block-wise matrix computation to prevent out-of-memory errors
MASTER_PORT=${MASTER_PORT:-29534}

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
GPUS_PER_NODE=${#GPU_ARRAY[@]}
CMD=(
  torchrun
  --nproc_per_node="${GPUS_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "/proposed/Ours/llama/main.py"
  --model_tag "${MODEL_TAG}"
  --model_path "${MODEL_PATH}"
  --dataset "${DATASET_TAG}"
  --i_block "${I_BLOCK}"
)

while true; do
  echo "[INFO] Launch: ${CMD[*]}"
  "${CMD[@]}"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[INFO] Script finished successfully (rc=0). Exiting..."
    break
  else
    echo "[ERROR] Script failed (rc=${rc}). Restarting in 2s..."
    sleep 2
  fi
done
