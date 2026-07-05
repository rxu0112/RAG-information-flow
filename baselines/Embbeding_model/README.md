# Embedding Model

## Before Use

Set defaults in `run_embedding.sh` or pass them as arguments.

## Run

```bash
bash baselines/Embbeding_model/run_embedding.sh <MODEL_TAG> <MODEL_PATH_PLACEHOLDER> <DATASET> <DEVICE>
```

| Argument | Choices |
|----------|---------|
| `MODEL_TAG` | `gemma-3-4B-it` / `Llama-3-8B-Instruct` / `Llama-3.2-3B-Instruct` |
| `MODEL_PATH_PLACEHOLDER` | e.g. `/path/to/Llama-3.2-3B-Instruct` |
| `DATASET` | `squad2` / `hotpot` / `msmarco` |
| `DEVICE` | `cuda:0`, `cuda:1`, ... |


