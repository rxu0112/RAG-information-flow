import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import json
import jsonlines
from vllm import LLM, SamplingParams
# import sys
# sys.path.append('../utils')
from Other_Baselines.ragu.utils.utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages, call_model
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.utils.utils import split_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="/Data/Info-flow_RAG/preprocessed_data/squad2_prepared.json")
parser.add_argument('--result_fp', type=str, default="output_data/SQuAD_generations.jsonl")
parser.add_argument('--model', type=str, default="Llama-3.2-3B-Instruct")
parser.add_argument('--prompt_name', type=str, default="chat_directRagQA_REAR3")
parser.add_argument("--num_generations", type=int, default=5,
            help="Number of generations to use")
parser.add_argument('--random_seed', type=int, default=10)
parser.add_argument("--dtype",  type=str, default=None,
            help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
parser.add_argument("--world_size",  type=int, default=1,
            help="world size to use multiple GPUs.")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--chat_template', action="store_true")      
parser.add_argument('--compute_pmi', action="store_true", default=False)  
parser.add_argument('--max_new_tokens', type=int, default=15)
parser.add_argument('--temperature', type=float, default=1.0,
            help="temperature at decoding. Zero means greedy sampling.")                        
parser.add_argument('--top_p', type=float, default=0.9,
            help="top-p sampling.")
parser.add_argument('--top_k', type=float, default=50,
            help="top-k sampling.")
parser.add_argument('--logprobs', type=int, default=1,
            help="number of log probs to return.")
parser.add_argument('--proportion', type=float, default=1.0,
            metavar='N', help='the proportion of data used')     
parser.add_argument('--do_stop', action="store_true", default=False)         

args = parser.parse_args()
def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    return input_data

def convert_squad_to_standard(input_file):
    # 读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        q_id = item.get("id", None)
        question = item.get("question", "")
        context = item.get("context", "")

        # 把所有答案的 text 收集到 golds
        answers = item.get("answers", [])
        if isinstance(answers, list):
            golds = [ans["text"] for ans in answers if "text" in ans]
        elif isinstance(answers, dict) and "text" in answers:
            golds = answers["text"]
        else:
            golds = []

        # 标准化后的样本
        converted.append({
            "q_id": q_id,
            "instruction": question,
            "paragraph": context,
            "golds": golds
        })
    return converted

    # 保存
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(converted, f, ensure_ascii=False, indent=2)

    # print(f"✅ 转换完成！保存到 {output_file} ，共 {len(converted)} 条数据")
device = 'cuda'

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(args.random_seed)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(args.random_seed)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(args.random_seed)

#Fix torch random seed
#torch.manual_seed(args.random_seed)

#enable_vllm_with_hidden_states()

if args.dtype is not None:
    model = LLM(model=args.model, dtype=args.dtype,
                tensor_parallel_size=args.world_size,) 
elif 'Llama-3.1-70B' in args.model and 'AWQ' in args.model:
    # https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html
    model = LLM(model=args.model, quantization='awq', enable_prefix_caching=True,
                tensor_parallel_size=args.world_size, trust_remote_code=True,
                gpu_memory_utilization=0.9, max_model_len=14336, enforce_eager=False) #download_dir=args.download_dir, awq_marlin                 
else:
    model = LLM(model=args.model, gpu_memory_utilization=0.9,
                tensor_parallel_size=args.world_size,)
tokenizer = model.get_tokenizer()

_prompt_name = args.prompt_name
if 'Llama' in args.model:
    _prompt_name += 'Llama'

# input_data = load_file(args.input_file)
# input_data = load_file(args.input_file)
# print('File uploaded, ', args.input_file, len(input_data))

input_data = convert_squad_to_standard(args.input_file)

def get_generations(model, input_data, number_of_generations):
    """For a given model, produce a number of generation """

    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if args.chat_template:
            processed_batch = []
            for item in batch:
                # By the time I'm writting this vLLM does not have the chat_template creation so we need to manually
                # do it. But concurently it seems that they are working on the vLLM to include this automatically?
                # see here: https://github.com/vllm-project/vllm/pull/6936
                # so keep to the current version w/ this code or watch out not to apply twice!!
                #https://huggingface.co/docs/transformers/chat_templating
                #https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
                tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model, _prompt_name, item), 
                                    tokenize=False, add_generation_prompt=True)
                processed_batch.append(tokenized_chat)

        else:
            processed_batch = [
                PROMPT_DICT[_prompt_name]['user'].format_map(item) for item in batch]
        for item in batch:
            del item["paragraph"] # just use this dict-key to re-use format
            del item["instruction"] # just use this dict-key to re-use format

        generations = []
        toklogprobs = []
        for i in range(number_of_generations):
            generation, toklogprob, _, _, _ = call_model(
                                    processed_batch, 
                                    model, 
                                    args,
                                    tokenizer,
                                    seed=i)
            generations.append(generation)
            toklogprobs.append(toklogprob)

        for j, item in enumerate(batch):
            item["prompt"] = processed_batch[j]
            item["generations"] = []
            for g in range(number_of_generations):
                gen = {
                    "generation": generations[g][j],
                    "toklogprobs": toklogprobs[g][j],
                }
                item["generations"].append(gen)

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        if args.chat_template:
            processed_batch = []
            for item in batch:
                # By the time I'm writting this vLLM does not have the chat_template creation so we need to manually
                # do it. But concurently it seems that they are working on the vLLM to include this automatically?
                # see here: https://github.com/vllm-project/vllm/pull/6936
                # so keep to the current version w/ this code or watch out not to apply twice!!
                #https://huggingface.co/docs/transformers/chat_templating
                #https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
                tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model, _prompt_name, item), 
                                    tokenize=False, add_generation_prompt=True)
                processed_batch.append(tokenized_chat)

        else:
            processed_batch = [
                PROMPT_DICT[_prompt_name].format_map(item) for item in batch]
        for item in batch:
            del item["paragraph"] # just use this dict-key to re-use format
            del item["instruction"] # just use this dict-key to re-use format

        generations = []
        toklogprobs = []
        for i in range(number_of_generations):

            generation, toklogprob, _, _, _ = call_model(
                                    processed_batch, 
                                    model, 
                                    args,
                                    tokenizer,
                                    seed=i)
            generations.append(generation)
            toklogprobs.append(toklogprob)
        
        for j, item in enumerate(batch):
            item["prompt"] = processed_batch[j]
            item["generations"] = []
            for g in range(number_of_generations):
                gen = {
                    "generation": generations[g][j],
                    "toklogprobs": toklogprobs[g][j],
                }
                item["generations"].append(gen)

    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to, ', args.result_fp)

    
get_generations(model, input_data, args.num_generations)