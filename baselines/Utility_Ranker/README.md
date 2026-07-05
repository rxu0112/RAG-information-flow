## Step 1: create_retrieval_data.py

```bash
python baselines/Utility_Ranker/data_creation/create_retrieval_data.py \
  --input_file results/{MODEL_TAG}_{DATASET}/{DATASET}_{SPLIT}_data.pt \
  --dataset {DATASET} \
  --context_path results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/context.jsonl \
  --output_file results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/absolute.jsonl
```

## Step 2: generate_passage_embeddings.py

Generate retrieval embeddings from `context.jsonl`:

```bash
python baselines/Utility_Ranker/contriever/generate_passage_embeddings.py \
  --passages results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/context.jsonl \
  --output_dir results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/embeddings \
  --model_name_or_path baselines/Utility_Ranker/contriever_msmarco
```

## Step 3: passage_retrieval.py

Run retrieval using the embeddings from Step 2:

```bash
python baselines/Utility_Ranker/contriever/passage_retrieval.py \
  --data results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/absolute.jsonl \
  --passages results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/context.jsonl \
  --passages_embeddings results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/embeddings/passages_* \
  --output_dir results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/retrieval \
  --model_name_or_path baselines/Utility_Ranker/contriever_msmarco \
  --n_docs 5
```

This step generates:

- `results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/retrieval/absolute_retrieval.jsonl`

## Step 4: utility_distill_run_llm.py

Run the LLM on the retrieval results:

```bash
python baselines/Utility_Ranker/retrieval_qa/utility_distill_run_llm.py \
  --model_name MODEL_PATH_PLACEHOLDER \
  --input_file results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/retrieval/absolute_retrieval.jsonl \
  --result_fp results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/utility_distill_run_llm.jsonl \
  --top_n 5 \
  --split train \
  --world_size 1
```

## Step 5: utility_distill_score.py

```bash
python baselines/Utility_Ranker/retrieval_qa/utility_distill_score.py \
  --input_file results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/utility_distill_run_llm.jsonl \
  --result_fp results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/distill_score.jsonl \
  --top_n 5
```

## Step 6: main.py

Before training and testing, make sure these three files exist:

- `results/{MODEL_TAG}_{DATASET}/baselines/temp/train/utility_ranker/distill_score.jsonl`
- `results/{MODEL_TAG}_{DATASET}/baselines/temp/val/utility_ranker/distill_score.jsonl`
- `results/{MODEL_TAG}_{DATASET}/baselines/temp/test/utility_ranker/distill_score.jsonl`

### Training

```bash
python baselines/Utility_Ranker/passage_utility/main.py \
  --input_file results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/distill_score.jsonl \
  --save_dir results/{MODEL_TAG}_{DATASET}/baselines/temp/train/utility_ranker/checkpoint \
  --top_n 5 \
  --do_train true \
  --do_test false
```

### Testing

```bash
python baselines/Utility_Ranker/passage_utility/main.py \
  --input_file results/{MODEL_TAG}_{DATASET}/baselines/temp/{SPLIT}/utility_ranker/distill_score.jsonl \
  --save_dir results/{MODEL_TAG}_{DATASET}/baselines/temp/test/utility_ranker/checkpoint \
  --top_n 5 \
  --utility_output results/{MODEL_TAG}_{DATASET}/baselines/temp/test/utility_ranker/utility_score.json \
  --score_output results/{MODEL_TAG}_{DATASET}/baselines/temp/test/utility_ranker/ranker_scores.json \
  --do_train false \
  --do_test true
```
