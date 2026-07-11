Collects all baseline scores into a single aligned JSON file.

## Baseline Methods

| Method | Type | Available Splits |
|--------|------|-----------------|
| `Attention_Score` | Inference-only (no training) | train / val / test |
| `Focus` | Inference-only (no training) | train / val / test |
| `Other_Baselines` | Inference-only (no training) | train / val / test |
| `Embbeding_model` | Training-based (XGBoost) | test only |
| `Utility_Ranker` | Training-based (BERT ranker) | test only |

> **Note**: `Attention_Score`, `Focus`, and `Other_Baselines` are inference-only — they can be run independently on each split, so train/val/test are all available. `Embbeding_model` and `Utility_Ranker` require a training phase (fitting a calibrator/ranker on train+val), so only the test split has meaningful final predictions.

## Run

### Auto-path mode (recommended)

```bash
python baselines/collect_baselines.py --model_tag <TAG> --dataset <DS> --split <SP>
```

| Argument | Description |
|----------|-------------|
| `--model_tag` | `gemma-3-4B-it` / `Llama-3-8B-Instruct` / `Llama-3.2-3B-Instruct` |
| `--dataset` | `squad2` / `hotpot` / `msmarco` |
| `--split` | `train` / `val` / `test` |

### Explicit paths

**Train / Val** — inference-only baselines (no `embed_score` or `utility_score`):

```bash
python baselines/collect_baselines.py \
  --uncertainty_scores path/to/uncertainty_scores.json \
  --attention_results  path/to/attention/results.json \
  --focus_results      path/to/focus/results.json \
  --out_path           path/to/output.json
```

**Test** — adds training-based baselines (`embed_score` + `utility_score`):

```bash
python baselines/collect_baselines.py \
  --uncertainty_scores path/to/uncertainty_scores.json \
  --attention_results  path/to/attention/results.json \
  --focus_results      path/to/focus/results.json \
  --embed_scores       path/to/embedding/embed_score.json \
  --utility_scores     path/to/utility_ranker/utility_score.json \
  --out_path           path/to/output.json
```
