import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import random
import torch
from vllm import LLM

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from Other_Baselines.ragu.utils.utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages, call_model

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="...")
parser.add_argument('--result_fp', type=str, default="...")
parser.add_argument('--model', type=str, default="MODEL_PATH_PLACEHOLDER")
parser.add_argument('--prompt_name', type=str, default="chat_directRagQA_REAR2")
parser.add_argument("--num_generations", type=int, default=10)
parser.add_argument('--random_seed', type=int, default=10)
parser.add_argument("--dtype", type=str, default=None)
parser.add_argument("--world_size", type=int, default=2)
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--chat_template', action="store_true")
parser.add_argument('--compute_pmi', action="store_true", default=False)
parser.add_argument('--max_new_tokens', type=int, default=10)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=float, default=50)
parser.add_argument('--logprobs', type=int, default=1)
parser.add_argument('--proportion', type=float, default=1.0)
parser.add_argument('--do_stop', action="store_true", default=False)

args = parser.parse_args()

os.environ['PYTHONHASHSEED'] = str(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)

model = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.world_size)
tokenizer = model.get_tokenizer()

_prompt_name = args.prompt_name
if 'Llama' in args.model:
    _prompt_name += 'Llama'
if 'gemma' in args.model:
    _prompt_name += 'gemma'


def convert_to_standard(input_file):
    """Convert .pt file to standardized QA format."""
    data_all = torch.load(input_file, weights_only=False)
    data = data_all['data']

    converted = []
    for item in data:
        answers = item.get("answers", [])
        if isinstance(answers, list):
            golds = [ans["text"] for ans in answers if "text" in ans]
        elif isinstance(answers, dict) and "text" in answers:
            golds = [answers["text"]]
        else:
            golds = []
        converted.append({
            "q_id": item.get("id", None),
            "instruction": item.get("question", ""),
            "paragraph": item.get("context", ""),
            "golds": golds,
        })
    return converted


def get_generations(model, input_data, number_of_generations):
    """For a given model, produce a number of generations per question."""

    num_batches = (len(input_data) + args.batch_size - 1) // args.batch_size
    for idx in tqdm(range(num_batches)):
        batch = input_data[idx * args.batch_size:(idx + 1) * args.batch_size]

        if args.chat_template:
            processed_batch = [
                tokenizer.apply_chat_template(
                    getChatMessages(args.model, _prompt_name, item),
                    tokenize=False, add_generation_prompt=True)
                for item in batch
            ]
        else:
            processed_batch = [
                PROMPT_DICT[_prompt_name]['user'].format_map(item) for item in batch
            ]

        for item in batch:
            item.pop("paragraph", None)
            item.pop("instruction", None)

        generations, toklogprobs = [], []
        for i in range(number_of_generations):
            generation, toklogprob, _, _, _ = call_model(
                processed_batch, model, args, tokenizer, seed=i)
            generations.append(generation)
            toklogprobs.append(toklogprob)

        for j, item in enumerate(batch):
            item["prompt"] = processed_batch[j]
            item["generations"] = []
            for g in range(number_of_generations):
                item["generations"].append({
                    "generation": generations[g][j],
                    "toklogprobs": toklogprobs[g][j],
                })

    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


input_data = convert_to_standard(args.input_file)
get_generations(model, input_data, args.num_generations)
