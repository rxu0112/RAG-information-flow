import json
import torch
from utils import *
import argparse

parser = argparse.ArgumentParser(description="faithfulness comparison")
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct','gemma-3-4B-it'], required=True)
parser.add_argument('--dataset', type=str, choices=['squad2','hotpot','msmarco'], required=True)

args = parser.parse_args()

with open(f'results/{args.model}_{args.dataset}/hem_faithfulness_bf16.json', 'r') as f:
    hem_faithfulness_list = json.load(f)

correct_wo_top_rank = [i for i, v in enumerate(hem_faithfulness_list) if v[0] < 0.5]
correct_wo_low_rank = [i for i, v in enumerate(hem_faithfulness_list) if v[1] > 0.5]

correct_wo_top_contri = [i for i, v in enumerate(hem_faithfulness_list) if v[2] < 0.5]
correct_wo_low_contri = [i for i, v in enumerate(hem_faithfulness_list) if v[3] > 0.5]

correct_wo_top_path = [i for i, v in enumerate(hem_faithfulness_list) if v[4] < 0.5]
correct_wo_low_path = [i for i, v in enumerate(hem_faithfulness_list) if v[5] > 0.5]