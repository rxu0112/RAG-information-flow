# Attention_Score

## Before Use

Set defaults in `run_Attention.sh` or pass them as arguments.

## Run

```bash
bash baselines/Attention_Score/run_Attention.sh <MODEL_TAG> <MODEL_PATH_PLACEHOLDER> <DATASET> <SPLIT> <GPU_ID>
```

| Argument | Choices |
|----------|---------|
| `MODEL_TAG` | `gemma-3-4B-it` / `Llama-3-8B-Instruct` / `Llama-3.2-3B-Instruct` |
| `MODEL_PATH_PLACEHOLDER` | e.g. `/path/to/Llama-3.2-3B-Instruct` |
| `DATASET` | `squad2` / `hotpot` / `msmarco` |
| `SPLIT` | `train` / `val` / `test` |
| `GPU_ID` | `0`, `1`, ... |
