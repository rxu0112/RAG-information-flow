import json
import torch
from utils import *
from rbo import rbo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'gemma-3-4b-it', 'Llama-3-8B-Instruct'], required=True)

args = parser.parse_args()

device_num = 8

with open(f"processed_data/{args.data}_prepared.json") as f:
    dataset_list = json.load(f)

ranking_list = []
contri_list = []
path_list = []
for i in range(device_num):
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_rank_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        rank = [json.loads(line) for line in f if line.strip()]
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_contri_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        contri = [json.loads(line) for line in f if line.strip()]
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_path_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        path = [json.loads(line) for line in f if line.strip()]
    ranking_list = ranking_list + rank
    contri_list = contri_list + contri
    path_list = path_list + path

with open(f'results/{args.model}_{args.dataset}/prediction_collection_bf16.json', 'r') as f:
    prediction_list = json.load(f)

with open(f'results/{args.model}_{args.dataset}/hem_answerable_collection_bf16.json', 'r') as f:
    hem_list = json.load(f)

indices_to_remove = torch.load(f'results/{args.model}_{args.dataset}/indices_to_remove.pt')

obtained_num = min([len(ranking_list), len(contri_list), len(path_list), len(prediction_list)])
dataset_list = dataset_list[0: obtained_num]
prediction_list = prediction_list[0: obtained_num]
hem_list = hem_list[0: obtained_num]

dataset_list = [v for i, v in enumerate(dataset_list) if i not in indices_to_remove]
prediction_list = [v for i, v in enumerate(prediction_list) if i not in indices_to_remove]
hem_list = [v for i, v in enumerate(hem_list) if i not in indices_to_remove]

labels = [1 if s > 0.5 else 0 for s in hem_list]
y = np.array(labels)

indices = np.arange(len(y))

temp_idx, test_idx, y_temp, y_test = train_test_split(
    indices, y, test_size=0.2, random_state=42)

train_idx, val_idx, y_train, y_val = train_test_split(
    temp_idx, y_temp, test_size=0.25, random_state=42)

hem_train = np.array(hem_list)[train_idx]
hem_val = np.array(hem_list)[val_idx]
hem_test = np.array(hem_list)[test_idx]

pred_train = [prediction_list[i] for i in train_idx]
pred_val = [prediction_list[i] for i in val_idx]
pred_test = [prediction_list[i] for i in test_idx]

dataset_train = [dataset_list[i] for i in train_idx]
dataset_val = [dataset_list[i] for i in val_idx]
dataset_test = [dataset_list[i] for i in test_idx]

test_summary = {"hem": hem_test, "pred": pred_test, "data": dataset_test}
train_summary = {"hem": hem_train, "pred": pred_train, "data": dataset_train}
val_summary = {"hem": hem_val, "pred": pred_val, "data": dataset_val}

torch.save(test_summary, f"results/{args.model}_{args.dataset}/{args.dataset}_test_data.pt")
torch.save(train_summary, f"results/{args.model}_{args.dataset}/{args.dataset}_train_data.pt")
torch.save(val_summary, f"results/{args.model}_{args.dataset}/{args.dataset}_val_data.pt")