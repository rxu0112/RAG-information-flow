#!/usr/bin/env bash
set -uo pipefail   

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,3
fi
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET
export TORCH_NCCL_DESYNC_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_TAG=${DATASET_TAG:-hotpot} # choose from squad2, hotpot, msmarco
MODEL_PATH=${MODEL_PATH:-MODEL_PATH_PLACEHOLDER}
I_BLOCK=${I_BLOCK:-100} # Block-wise matrix computation to prevent out-of-memory errors
MASTER_PORT=${MASTER_PORT:-29536}

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
GPUS_PER_NODE=${#GPU_ARRAY[@]}

CMD=(
  torchrun
  --nproc_per_node="${GPUS_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "/proposed/Ours/gemma/main.py"
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
