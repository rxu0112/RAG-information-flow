import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility_Ranker.data_creation.read_dataset import get_entry_from_dataset
from Utility_Ranker.utils.utils import load_jsonlines, save_file_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--dataset', default='SQuAD')
    parser.add_argument(
        '--context_path',
        default="",
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="",
    )
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

    os.makedirs(Path(args.context_path).parent, exist_ok=True)
    os.makedirs(Path(args.output_file).parent, exist_ok=True)

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
