import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM
import sys
sys.path.append('../utils')
import os
# sys.path.append()
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  
from Other_Baselines.ragu.utils.utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages, call_model
from Other_Baselines.ragu.retrieval_qa.metrics import metric_max_over_ground_truths, exact_match_score, match, f1
import evaluate

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="、gemma-3-4b-it")
    parser.add_argument('--input_file', type=str, default='ragu/new_data/hotpot/gemma_test_retrieval_output.jsonl')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str, default='ragu/new_data/hotpot/gemma_test_utility_distil_run_llm.jsonl')
    parser.add_argument('--prompt_name', type=str, default="chat_directRagQA_REAR2")
    parser.add_argument('--prompt_name_cb', type=str, default="prompt_noRAG_RECOMP")
    parser.add_argument('--batch_size', type=int, default=800)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=4,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--chat_template', action="store_true")
    parser.add_argument('--fewshots', type=str, help="list of examples used for few-shot prompting --jsonl file",
                        default=None)      
    parser.add_argument('--split', type=str, help="split to process", default=None)                         
    parser.add_argument('--compute_pmi', action="store_true")   # this should be False, we dont do pmi in distillation phase

    # sampling params
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="temperature at decoding. Zero means greedy sampling.")                        
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="top-p sampling.")
    parser.add_argument('--top_k', type=int, default=-1,
                    help="top-k sampling.")
    parser.add_argument('--logprobs', type=int, default=1,
                    help="number of log probs to return.")       
    parser.add_argument('--proportion', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to train')       
    parser.add_argument('--proportion_dev', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to dev')       
    parser.add_argument('--proportion_test', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to test')   
    parser.add_argument('--do_stop', action="store_true", default=False)
                            
    
    args = parser.parse_args()

    if args.dtype is not None:
        model = LLM(model=args.model_name, dtype=args.dtype,
                    tensor_parallel_size=args.world_size,) #download_dir=args.download_dir,
    elif 'Llama-3.1-70B' in args.model_name and 'AWQ' in args.model_name:
        # https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html
        model = LLM(model=args.model_name, quantization='awq', enable_prefix_caching=True,
                    tensor_parallel_size=args.world_size, trust_remote_code=True,
                    gpu_memory_utilization=0.9, max_model_len=14336, enforce_eager=False) #download_dir=args.download_dir, awq_marlin                    
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size,) # download_dir=args.download_dir,
    tokenizer = model.get_tokenizer()

    input_data = load_file(args.input_file)
    print(len(input_data))
    print('File uploaded, ', args.input_file, len(input_data))

    _prompt_name = args.prompt_name
    _prompt_name_cb = args.prompt_name_cb
    if 'Llama' in args.model_name:
        _prompt_name += 'Llama'
        # _prompt_name_cb += 'Llama'


    # if using few shot training examples, remove them from the training, create the prompt string
    if args.fewshots:
        fewshots = load_file(args.fewshots)
        if args.split == 'train':
            fewshots_ids = [x['q_id'] for x in fewshots]
            input_data = [x for x in input_data if not x['q_id'] in fewshots_ids]
            print('Fewshots removed, ', len(input_data))
        fewshot_prompt = "".join(['{}\nAnswer: {}\n\n'.format(x['question'],x['answers'][0]) for x in fewshots])

    for id, item in enumerate(input_data):
        retrieval_result = item["ctxs"][:args.top_n]
        evidences = ["[1] " + ctx["text"] for i, ctx in enumerate(retrieval_result)]
        item["paragraphs"] = evidences

        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
            print(item["instruction"] + '\n')
        
        if args.fewshots:
            item['fewshots'] = fewshot_prompt
    
    data_proportion = 1.0
    if args.split == 'train':
        data_proportion = args.proportion
    elif args.split =='dev':
        data_proportion = args.proportion_dev
    else:
        data_proportion = args.proportion_test

    # print(f'Processing only the first {data_proportion}% of the data.')
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = []
        for item in batch:
            for evidence in item["paragraphs"]: # number of evidences will be args.top_n
                item["paragraph"] = evidence
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, _prompt_name, item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT[_prompt_name]['user'].format_map(item))

            # add llm call with closed-book call prompt_no_input_SELFRAG yields long respones
            if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, _prompt_name, item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)                
            else:
                processed_batch.append(PROMPT_DICT[_prompt_name_cb]['user'].format_map(item))
            del item["paragraph"] # just use this dict-key to re-use format
            del item["paragraphs"] # just use this dict-key to re-use format
            del item["instruction"] # just use this dict-key to re-use format

        preds, toklogprobs, _, _, _ = call_model(processed_batch, model, args, tokenizer)

        l = 0
        for j, item in enumerate(batch):
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                ctx["output"] = preds[l]
                ctx["toklogprob"] = toklogprobs[l]
                l +=1
            item["closed-book"] ={
                "output": preds[l],
                "toklogprob": toklogprobs[l],
                }
            l +=1

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        processed_batch = []
        for item in batch:
            for evidence in item["paragraphs"]: # number of evidences will be args.top_n
                item["paragraph"] = evidence
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, _prompt_name, item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT[_prompt_name]['user'].format_map(item))

            # add llm call with closed-book call prompt_no_input_SELFRAG yields long respones
            if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, _prompt_name, item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)                
            else:
                processed_batch.append(PROMPT_DICT[_prompt_name_cb]['user'].format_map(item))
            del item["paragraph"] # just use this dict-key to re-use format
            del item["paragraphs"] # just use this dict-key to re-use format
            del item["instruction"] # just use this dict-key to re-use format
        
        preds, toklogprobs, _, _, _ = call_model(processed_batch, model, args, tokenizer)

        l = 0
        for j, item in enumerate(batch):
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                ctx["output"] = preds[l]
                ctx["toklogprob"] = toklogprobs[l]
                l +=1
            item["closed-book"] ={
            "output": preds[l],
            "toklogprob": toklogprobs[l],
            }
            l +=1

    print('Finalised QA, start evaluation...')
    #rouge = evaluate.load('rouge')

    print(f'Evaluate only the first {data_proportion}% of the data.')
    for item in input_data[:int(len(input_data)*data_proportion)]:
        for i, ctx in enumerate(item["ctxs"][:args.top_n]):
            if ctx["output"]:
            #    ctx["EM"] = metric_max_over_ground_truths(
            #         exact_match_score, ctx["output"], item["golds"])
               ctx["F1"] = f1([ctx["output"]], item["golds"])
               ctx["acc"] = match(ctx["output"], item["golds"])
               ctx["nll"] = (-sum(ctx["toklogprob"]))/len(ctx["toklogprob"]) if len(ctx["toklogprob"])>0 else 0
               ctx["ppl"] = np.exp(ctx["nll"])
               ctx["rougeL"] = 0 # max([rouge.compute(predictions=[ctx["output"]], references=[answer])["rougeL"]
               #                     for answer in item["golds"]]) ## no need .mid.fmeasure as using aggregator and returning fmeasure
            else:
                # ctx["EM"] = 0
                ctx["F1"] = 0
                ctx["acc"] = 0
                ctx["nll"] = 0
                ctx["ppl"] = 0
                ctx["rougeL"] = 0
        # eval closed-book qa
        if item["closed-book"]["output"]:
           item["closed-book"]["acc"] = match(item["closed-book"]["output"], item["golds"])
           item["closed-book"]["nll"] = (-sum(item["closed-book"]["toklogprob"]))/len(item["closed-book"]["toklogprob"]) if len(item["closed-book"]["toklogprob"])>0 else 0
           item["closed-book"]["ppl"] = np.exp(item["closed-book"]["nll"])
           item["closed-book"]["rougeL"] = 0 #max([rouge.compute(predictions=[item["closed-book"]["output"]], references=[answer])["rougeL"]
           #                         for answer in item["golds"]])
        else:
            item["closed-book"]["acc"] = 0
            item["closed-book"]["nll"] = 0
            item["closed-book"]["ppl"] = 0
            item["closed-book"]["rougeL"] = 0


    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


if __name__ == "__main__":
    main()