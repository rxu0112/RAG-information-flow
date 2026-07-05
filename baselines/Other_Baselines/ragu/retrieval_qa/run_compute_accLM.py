import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import sys
sys.path.append('../utils')
from Other_Baselines.ragu.utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages
from Other_Baselines.ragu.retrieval_qa.metrics import match


import vllm.envs as envs 
print("envs.VLLM_ATTENTION_BACKEND: " + str(envs.VLLM_ATTENTION_BACKEND)) 


def call_model(prompts, model, warmup_caching=None):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=5, logprobs=1)
    if warmup_caching==0:
        print('warming up...')
        out = model.generate(prompts[0], sampling_params, use_tqdm=False)
        print(out[0].outputs[0].text, 'warm-up done.')
    preds = model.generate(prompts, sampling_params, use_tqdm=False)

    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    return postprocessed_preds


def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--eval_distil', action="store_true")
    parser.add_argument('--acc', action="store_true", help="recompute rule based accuracy")
    parser.add_argument('--prompt_name', type=str, default="prompt_accuracy_eval", 
                        help="other value is: 'chat_accuracy_eval-rlhf-calib' used with chat_template")
    parser.add_argument('--chat_template', action="store_true")
    parser.add_argument("--shard_size",  type=int, default=None)
    parser.add_argument("--shard_id",  type=int, default=None)
    
    args = parser.parse_args()

    if args.dtype is not None:
        model = LLM(model=args.model_name, dtype=args.dtype,
                    tensor_parallel_size=args.world_size,) #download_dir=args.download_dir,
    elif 'Qwen2-72B' in args.model_name and 'AWQ' in args.model_name:
        # https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html
        model = LLM(model=args.model_name, quantization='awq', enable_prefix_caching=False,
                    tensor_parallel_size=args.world_size, trust_remote_code=True,
                    gpu_memory_utilization=0.9, max_model_len=10336, enforce_eager=False) #download_dir=args.download_dir, awq_marlin # max_model_len=14336
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size,) # download_dir=args.download_dir,
    
    tokenizer = model.get_tokenizer()

    input_data = load_file(args.input_file)
    print('File uploaded, ', args.input_file, len(input_data),'distill QA' if args.eval_distil else 'RAG QA')
    print('Formatting input data...')
    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

    if args.shard_id != None and args.shard_size != None:
        xs = list(range(len(input_data)))
        shards = xs[0::args.shard_size]
        if args.shard_id == len(shards):
            input_data = input_data[shards[args.shard_id-1]:]
        elif args.shard_id > len(shards):
            print('No such shard id.')
            exit(0)
        else:
            input_data = input_data[shards[args.shard_id-1]:shards[args.shard_id]]
        print(f"Inference on shard {args.shard_id}, size {len(input_data)}")

    print('Start annotation...')
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = []
        if args.eval_distil:
            for item in batch:
                for ctx in item["ctxs"][:args.top_n]: # number of evidences will be args.top_n
                    eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': ctx['output']}
                    if args.chat_template:
                        tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                    tokenize=False, add_generation_prompt=True)
                        processed_batch.append(tokenized_chat)
                    else:
                        processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))
                # add llm call with closed-book 
                eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': item['closed-book']['output']}
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                    tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))
        else:
            for item in batch:
                eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': item['output']}
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))

        preds = call_model(processed_batch, model=model, warmup_caching=idx)

        preds = [0 if ('incorrect' in pred) or not ('correct' in pred) else 1 for pred in preds]
        
        if args.eval_distil:
            l = 0
            for j, item in enumerate(batch):
                for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                    ctx["acc_LM"] = preds[l]
                    l +=1
                    if args.acc:
                        ctx["acc"] = match(ctx["output"], item["golds"])
                item["closed-book"]["acc_LM"] = preds[l]
                l +=1
                if args.acc:
                    item["closed-book"]["acc"] = match(item["closed-book"]["output"], item["golds"])
        else:
            for j, item in enumerate(batch):
                item["acc_LM"] = preds[j]
                if args.acc:
                    item["acc"] = match(item["output"], item["golds"])


    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        processed_batch = []
        if args.eval_distil:
            for item in batch:
                for ctx in item["ctxs"][:args.top_n]: # number of evidences will be args.top_n
                    eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': ctx['output']}
                    if args.chat_template:
                        tokenized_chat = tokenizer.apply_chat_template(
                                            getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                            tokenize=False, add_generation_prompt=True)
                        processed_batch.append(tokenized_chat)
                    else:
                        processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))
                # add llm call with closed-book call prompt_no_input_SELFRAG yields long respones
                eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': item['closed-book']['output']}
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))
        else:
            for item in batch:
                eval_item = {'instruction': item['instruction'], 'answers': item['golds'], 'output': item['output']}
                if args.chat_template:
                    tokenized_chat = tokenizer.apply_chat_template(
                                        getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                        tokenize=False, add_generation_prompt=True)
                    processed_batch.append(tokenized_chat)
                else:
                    processed_batch.append(PROMPT_DICT["prompt_accuracy_eval"].format_map(eval_item))
        
        preds = call_model(processed_batch, model=model)

        preds = [0 if ('incorrect' in pred) or not ('correct' in pred) else 1 for pred in preds]                

        if args.eval_distil:
            l = 0
            for j, item in enumerate(batch):
                for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                    ctx["acc_LM"] = preds[l]
                    l +=1
                    if args.acc:
                        ctx["acc"] = match(ctx["output"], item["golds"])
                item["closed-book"]["acc_LM"] = preds[l]
                l +=1
                if args.acc:
                    item["closed-book"]["acc"] = match(item["closed-book"]["output"], item["golds"])
        else:
            for j, item in enumerate(batch):
                item["acc_LM"] = preds[j]
                if args.acc:
                    item["acc"] = match(item["output"], item["golds"])

    print('Finalised QA, start evaluation...')


    if args.eval_distil:
        lines = []
        if args.shard_id != None:
            lines.append('Process-{}'.format(args.shard_id))
        lines.append('Per context QA ...')
        lines.append("overall result acc_LM: {0}".format(
            np.mean([ctx["acc_LM"] for item in input_data for ctx in item["ctxs"][:args.top_n]])))
        lines.append("overall result acc: {0}".format(
            np.mean([ctx["acc"] for item in input_data for ctx in item["ctxs"][:args.top_n]])))
        lines.append('Closed-book QA ...')
        lines.append("overall result acc_LM: {0}".format(
            np.mean([item["closed-book"]["acc_LM"] for item in input_data])))      
        lines.append("overall result acc: {0}".format(
            np.mean([item["closed-book"]["acc"] for item in input_data])))           
        lines.append("\n".join(lines))            
    else:
        print('RAG QA top:',args.top_n,' ...')
        print("overall result acc_LM: {0}".format(
            np.mean([item["acc_LM"] for item in input_data])))      
        print("overall result acc: {0}".format(
            np.mean([item["acc"] for item in input_data])))      

    if args.shard_id != None:
        save_file_jsonl(input_data, args.result_fp.replace('.jsonl', '-sh{}.jsonl'.format(args.shard_id)))
        print('Files saved to, ', args.result_fp.replace('.jsonl', '-sh{}.jsonl'.format(args.shard_id)))    
    else:
        save_file_jsonl(input_data, args.result_fp)
        print('Files saved to, ', args.result_fp)

if __name__ == "__main__":
    main()