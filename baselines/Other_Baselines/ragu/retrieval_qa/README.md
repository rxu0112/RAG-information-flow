### Retrieval Augmented Answer Generation

To generate answers to questions with a set of (```TOP```) retrieved passages as context you should use the following script:

```
python retrieval_qa/run_baseline_lm.py
   --model_name $LLM
   --split $SPLIT
   --input_file $HOMEDATA/$DATASET-$SPLIT.jsonl
   --result_fp $HOMEOUT/$MODEL-$DATASET-$SPLIT-RAGQAt${TOP}_ctREAR3.jsonl
   --prompt_name \"chat_directRagQA_REAR3\"
   --chat_template
   --top_n $TOP
   --temperature 0.0
   --top_p 1
   --max_new_tokens 50
   --do_stop
   --logprobs 1
   --compute_pmi
```

```--input_file``` specifies a .jsonl file as created [here](../data_creation/README.md) (i.e., question, retrieved passages, and gold answers).
```--input_file``` will be the same as the input file but with several additional fields with information computed in this script (e.g., generated most likely answer, token log-probs, etc.).  
```--chat_template``` in our experiments we use chat LLMs. The prompt ```chat_directRagQA_REAR3``` is defined in [utils.py](utils.py)

If you want to generate training data required for p(true) few-shots then you should specify ```SPLIT="train"``` and the following flags:
```  
  --p_true_sample
  --p_true_num_fewshot 20
  --proportion 1
```

#### Generate target QA Model (LLM) judgements

```
python3 retrieval_qa/utility_distill_run_llm.py 
   --model_name $LLM 
   --input_file $HOMEDATA/$data-$split.jsonl 
   --result_fp $HOMEOUT/$model-$data-$split-distil_t${TOP}_ctREAR3.jsonl
   --split $SPLIT
   --prompt_name \"chat_directRagQA_REAR3\" 
   --prompt_name_cb \"chat_noRAG_REAR3\"
   --chat_template
   --top_n $TOP 
   --temperature 0.0 
   --top_p 1 
   --max_new_tokens 50 
   --do_stop
```

```prompt_name_cb```here closed-book generations are also stored and use this template.

Once we get LLM-judgements, we also add Natural Langauge Inference annotations with the following script:

```
python3 retrieval_qa/utility_distill_score.py
   --input_file $HOMEOUT/$model-$data-$split-distil_t${TOP}_ctREAR3.jsonl
   --result_fp  $HOMEOUT/$model-$data-$split-distil_t${TOP}_ctREAR3-score.jsonl
   --top_n $TOP
```

Finally we run the LLM-based Accuracy Evaluator (see below) with the flag ```--eval_distil``` on.


### LLM-based Accuracy Evaluator

We use [Qwen/Qwen2-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-72B-Instruct-AWQ) as our accuracy evaluator.
With the prompt proposed in [(Sun et. al, 2024)](https://aclanthology.org/2024.naacl-long.18.pdf). The prompt is called ```prompt_accuracy_eval``` and can be found in the prompt dictionary [utils.py](./utils.py).

The script to evaluate QA generated answers is the following. The input file is a output by the answer generation script (either RAG QA or m). The output is the same .jsonl file with the field ```--acc_LM``` (0/1 values) added to each example:

```
python retrieval_qa/run_compute_accLM.py  
  --model_name Qwen/Qwen2-72B-Instruct-AWQ  
  --input_file JSONL_WITH_GENERATED_ANSWERS  
  --result_fp  JSONL_ACCLM_ANNOTATED_ANSWERS
  --acc 
  --top_n TOP 
  --eval_distil 
```

```--acc``` indicates to also copute token based accuracy (i.e., as whether the gold answer is contained in the generated answer, e.g., [(Asai et al., 2024)](https://openreview.net/forum?id=hSyW5go0v8)).  
```--eval_distil``` indicates whether to evaluate an input file with generated answers for each of the ```TOP``` input passages. If this flag is not set, then, it will evaluate an input file with answers generated for each questions with retrieval augmented QA (```TOP``` does not need to be specified in this case).


### Inference Library

We use the [vLLM library](https://docs.vllm.ai/en/latest/) for faster inference, and you can load the model like this:

```
model = LLM(model=args.model_name, quantization='awq', enable_prefix_caching=True,
            tensor_parallel_size=args.world_size, trust_remote_code=True,
            gpu_memory_utilization=0.9, max_model_len=14336, enforce_eager=False) 
```
            
Note: We didnt use awq_marlin for quantization as somewhere seemed to be an issue.

