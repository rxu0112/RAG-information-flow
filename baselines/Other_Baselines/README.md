# Other_Baselines

## Before Use

Set defaults in both scripts or pass them as arguments.

## Run

```bash
# Step 1: multi-round generation (vLLM)
bash baselines/Other_Baselines/scripts/run_step1_generate.sh <MODEL_TAG> <MODEL_PATH_PLACEHOLDER> <DATASET> <SPLIT>

# Step 2: compute uncertainty scores
bash baselines/Other_Baselines/scripts/run_step2_uncertainty.sh <MODEL_TAG> <MODEL_PATH_PLACEHOLDER> <DATASET> <SPLIT>
```

| Argument | Choices |
|----------|---------|
| `MODEL_TAG` | `gemma-3-4B-it` / `Llama-3-8B-Instruct` / `Llama-3.2-3B-Instruct` |
| `MODEL_PATH_PLACEHOLDER` | e.g. `/path/to/Llama-3.2-3B-Instruct` |
| `DATASET` | `squad2` / `hotpot` / `msmarco` |
| `SPLIT` | `train` / `val` / `test` |

