from tqdm import tqdm
import jsonlines
import argparse
import json
import torch
import os
import torch.multiprocessing as mp
print("Current file:", __file__)
print("Current dir:", os.path.dirname(__file__))
import csv
from Other_Baselines.ragu.data_creation.read_dataset import get_entry_from_dataset
import sys
from Other_Baselines.ragu.utils.utils import load_jsonlines, save_file_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='...')
    parser.add_argument('--dataset', default='SQuAD')
    parser.add_argument('--context_path', default='./context.jsonl')
    parser.add_argument('--output_file', type=str, default='...')
    args = parser.parse_args()
    processed_data = []

    if args.input_file.endswith(".json"):
        data = json.load(open(args.input_file))
    elif args.input_file.endswith(".jsonl"):
        data = load_jsonlines(args.input_file)
    elif args.input_file.endswith(".pt"):
        data = torch.load(args.input_file, weights_only=False)
        data = data['data']

    if args.dataset in ['NQ', 'TQA']:
        data = data['data']
    save_path = args.context_path
    with open(save_path, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(data)):
            ctx = item['context']
            processed_data.append(get_entry_from_dataset(args.dataset, item, idx))
            item = {"context": ctx, "id": idx}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(len(processed_data))
    save_file_jsonl(processed_data, args.output_file)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
