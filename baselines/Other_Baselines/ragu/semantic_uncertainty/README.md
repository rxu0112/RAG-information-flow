Original instructions for the base code can be found following this link [Semantic Entropy](https://github.com/jlko/semantic_uncertainty). Below specific details of our use in this project.

### Generate Samples

We run the following command to generate the samples for those approaches that use sampling.

```
python semantic_uncertainty/generate.py
   --model $LLM
   --input_file $HOMEDATA/$data-$split.jsonl 
   --result_fp $HOMEOUT/$model-$data-$split-samples_t${TOP}_$prompt.jsonl
   --prompt_name \"chat_directRagQA_REAR3\"
   --chat_template
   --top_n $TOP
   --split $SPLIT
   --max_new_tokens 50
   --do_stop
   --num_generations 10
```

Example values are: 
```
data="nq"
split="dev"
model="gemma-2-2b-it"
TOP="5"
prompt="ctREAR3"
"llm="google/gemma-2-2b-it"
```

If we specify split="train" and the flags below we can generate the train samples required later for the p(true) approach:

```
   --p_true_sample
   --p_true_num_fewshot 20
   --proportion 0.5
```

### Compute Uncertainty Metrics and Eval

To compute other uncertainty estimation approaches and final metrics, we can run the command below.
It will compute Semantic Entropy, Cluster Assignment and run uncertainty estimation evaluation (AUROC, AURAC).

```
python3 semantic_uncertainty/generate_answers.py
   --model_name $llm
   --precomputed_gen
   --no-get_training_set_generations
   --no-get_training_set_generations_most_likely_only
   --dataset $data
   --most_likely_file $HOMEOUT/$model-$data-SPLIT-RAGQAt${TOP}_$prompt-qwen2-72b-it-awq.jsonl
   --top_n $TOP
   --samples_file $HOMEOUT/$model-$data-SPLIT-samples_t${TOP}_$prompt.jsonl
   --utilities_file $HOMEOUT/$model-$data-$split-distil_t${TOP}_$prompt-score-qwen2-72b-it-awq-point_1.0acc_LM-nli_1.00be_combined_pred.jsonl
   --original_file $HOMEDATA/$data-SPLIT.jsonl
   --acc_LM
   --model_max_new_tokens 50
   --eval_mode $split
```

```--most_likely_file``` is the file containing the greedy answer generations for RAG with top ```TOP``` passages. 
```--samples_file``` is the file containing the N sampled answers.
```--utilities_file``` is the file containing the annotations from the Passage Utility predictor (see [passage utility/](../passage_utility/)).
```--original_file``` is the file containing all input data (question, gold answers, retrieved passages, etc. see [data_creation/](../data_creation/)).

