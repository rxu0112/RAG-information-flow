import argparse
import json
import os
from pathlib import Path

import heapq
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from Ours.llama.llama import AttrConfig, LlamaAttributor


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_CHOICES = ("squad2", "hotpot", "msmarco")
MODEL_TAG_CHOICES = ("Llama-3-8B-Instruct", "Llama-3.2-3B-Instruct")
DTYPE_NAME = "bf16"
METRIC_NAME = "manhattan"
DATASET_SAMPLE_LIMITS = {
    "squad2": 50000,
    "msmarco": 45000,
    "hotpot": 40000,
}


class OrderedSet:
    def __init__(self):
        self._data = {}

    def add(self, item):
        self._data[item] = None

    def __iter__(self):
        return iter(self._data.keys())

    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"OrderedSet({list(self._data.keys())})"


def _load_seen_ids(jsonl_path: str):
    seen = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "id" in obj:
                    seen.add(obj["id"])
    return seen


def _data_path(dataset_name: str) -> Path:
    return REPO_ROOT / "preprocessed_data" / f"{dataset_name}_prepared.json"


def _load_local_examples(path: Path):
    return load_dataset("json", data_files={"validation": str(path)})["validation"]


def _limit_dataset_examples(dataset, dataset_name: str):
    limit = DATASET_SAMPLE_LIMITS.get(dataset_name)
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def compute_ranking_from_layer_mats(device, matrices_arrays):
    assert len(matrices_arrays) > 0, "empty matrices_arrays"
    mats = []
    for matrix in matrices_arrays:
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().float().cpu().numpy()
        matrix = np.asarray(matrix)
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "each layer must be (T,T)"
        mats.append(matrix.astype(np.float32, copy=False))

    seq_len = mats[0].shape[0]
    for matrix in mats:
        assert matrix.shape == (seq_len, seq_len), "all layer mats must be the same shape"

    recorded_nodes = OrderedSet()
    start = seq_len - 1
    recorded_nodes.add(start)

    candidates = {str(start): [arr[start, :start].copy() for arr in mats]}

    extracted_matrices = [np.zeros((seq_len, seq_len), dtype=np.float32) for _ in range(len(mats))]
    for idx in range(len(extracted_matrices)):
        extracted_matrices[idx][-1, -1] = mats[idx][-1, -1]

    heap = []
    for group, arrays in candidates.items():
        for arr_idx, arr in enumerate(arrays):
            for pos_idx, value in enumerate(arr):
                heapq.heappush(heap, (-float(value), group, arr_idx, pos_idx))

    while len(recorded_nodes) < start + 1:
        if not heap:
            for pos_idx in range(start - 1, -1, -1):
                if pos_idx not in recorded_nodes:
                    recorded_nodes.add(pos_idx)
            break

        _, group, arr_idx, pos_idx = heapq.heappop(heap)
        recorded_nodes.add(pos_idx)

        group_idx = int(group)
        extracted_matrices[arr_idx][group_idx, pos_idx] = mats[arr_idx][group_idx, pos_idx]
        for prev_idx, matrix in enumerate(extracted_matrices[:arr_idx]):
            matrix[pos_idx, pos_idx] = mats[prev_idx][pos_idx, pos_idx]

        key = str(pos_idx)
        if key not in candidates:
            candidates[key] = []

        old_len = len(candidates[key])
        if arr_idx > old_len:
            new_part = [arr[pos_idx, :pos_idx].copy() for arr in mats[old_len:arr_idx]]
            candidates[key].extend(new_part)

            for new_idx, arr in enumerate(new_part, start=old_len):
                for new_pos_idx, value in enumerate(arr):
                    heapq.heappush(heap, (-float(value), key, new_idx, new_pos_idx))

    ranking = list(recorded_nodes)

    product = torch.from_numpy(extracted_matrices[0]).to(device=device, dtype=torch.float32)
    for matrix in extracted_matrices[1:]:
        matrix_tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
        product = torch.matmul(matrix_tensor, product)
    last_row = product[-1, :]
    return ranking, last_row.cpu().tolist()


def loop_result(tool, device: str, prompt: str):
    encoded = tool.tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    _ = [tool.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_ids[0].tolist()]

    contrib_per_step = []
    rankings_per_step = []
    path_per_step = []

    for _ in range(10):
        next_id, next_text = tool.generate_one_token(
            prompt, do_sample=False, temperature=1.0, top_p=1.0
        )

        if tool.tokenizer.eos_token_id is not None and next_id == tool.tokenizer.eos_token_id:
            break
        if next_text.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>"}:
            break

        encoded = tool.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        _, seq_len = input_ids.shape

        oproj_refs = tool._capture_o_proj_refs(encoded)
        contrib_mats = tool._manual_forward_once_and_calc_matrices(input_ids, oproj_refs)
        contrib = tool._accumulate_last_token(device, contrib_mats, seq_len)

        prompt += next_text
        contrib_per_step.append(contrib)

        mats_np = []
        for matrix in contrib_mats:
            if hasattr(matrix, "detach"):
                mats_np.append(matrix.detach().float().cpu().numpy())
            else:
                mats_np.append(np.asarray(matrix, dtype=np.float16))

        ranking, last_path = compute_ranking_from_layer_mats(device, mats_np)
        rankings_per_step.append(ranking)
        path_per_step.append(last_path)

    return prompt, contrib_per_step, rankings_per_step, path_per_step


def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def shard_indices_contiguous(n_items: int, world_size: int, rank: int):
    base = n_items // world_size
    remainder = n_items % world_size

    if rank < remainder:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base

    return list(range(start, min(end, n_items)))


def main():
    parser = argparse.ArgumentParser(
        description="Compute token contribution paths for Llama models in distributed mode."
    )
    parser.add_argument("--model_tag", type=str, choices=MODEL_TAG_CHOICES, default="Llama-3.2-3B-Instruct")
    parser.add_argument("--model_path", type=str, default="MODEL_PATH_PLACEHOLDER") # path of args.model
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--i_block", type=int, default=100) # Number of rows processed per block when computing contribution matrices; smaller values reduce peak GPU memory usage but usually make the run slower.
    args = parser.parse_args()
    args = parser.parse_args()

    model_tag = args.model_tag
    model_path = args.model_path
    resolved_data_path = _data_path(args.dataset)
    output_dir = REPO_ROOT / "results" / f"{model_tag}_{args.dataset}" / f"loop_{DTYPE_NAME}"

    rank, local_rank, world_size = init_dist()
    device = f"cuda:{local_rank}"
    torch.set_grad_enabled(False)

    if rank == 0:
        print(
            f"[INFO] world_size={world_size} | model={model_tag} | model_path={model_path} "
            f"| data_path={resolved_data_path} | dtype={DTYPE_NAME} | metric={METRIC_NAME}"
        )

    dataset = _load_local_examples(resolved_data_path)
    dataset = _limit_dataset_examples(dataset, args.dataset)
    n_all = len(dataset)

    idxs = shard_indices_contiguous(n_all, world_size, rank)
    shard = dataset.select(idxs)

    if rank == 0:
        counts = [len(shard_indices_contiguous(n_all, world_size, worker_rank)) for worker_rank in range(world_size)]
        print(f"[INFO] dataset={n_all}, per-division counts={counts}")

    cfg = AttrConfig(
        model_path=model_path,
        device=device,
        i_block=args.i_block,
    )
    tool = LlamaAttributor(cfg)

    contri_jsonl = str(output_dir / f"loop_{METRIC_NAME}_contri_{DTYPE_NAME}_{rank}.jsonl")
    rankings_jsonl = str(output_dir / f"loop_{METRIC_NAME}_rank_{DTYPE_NAME}_{rank}.jsonl")
    path_jsonl = str(output_dir / f"loop_{METRIC_NAME}_path_{DTYPE_NAME}_{rank}.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)

    seen_contri = _load_seen_ids(contri_jsonl)
    seen_rankings = _load_seen_ids(rankings_jsonl)
    seen_path = _load_seen_ids(path_jsonl)

    handles = {
        "contri_f": open(contri_jsonl, "a", encoding="utf-8", buffering=1),
        "rankings_f": open(rankings_jsonl, "a", encoding="utf-8", buffering=1),
        "path_f": open(path_jsonl, "a", encoding="utf-8", buffering=1),
    }

    pbar = tqdm(range(len(shard)), desc=f"Division{rank} (GPU {local_rank})", disable=(rank != 0))
    for shard_idx in pbar:
        example = shard[shard_idx]
        ex_id = example.get("id")
        if ex_id is None:
            ex_id = f"division{rank}_idx{shard_idx}"

        context = example["context"]
        question = example["question"]
        prompt = tool.build_prompt(context, question)

        if ex_id in seen_contri and ex_id in seen_rankings and ex_id in seen_path:
            continue

        input_output_seq, contri_loop, ranking_example, path_steps = loop_result(
            tool=tool, device=device, prompt=prompt
        )

        if ex_id not in seen_contri:
            record = {"id": ex_id, "seq": input_output_seq, "contri": contri_loop}
            handles["contri_f"].write(json.dumps(record, ensure_ascii=False) + "\n")
            handles["contri_f"].flush()
            seen_contri.add(ex_id)

        if ex_id not in seen_path:
            record = {"id": ex_id, "seq": input_output_seq, "path": path_steps}
            handles["path_f"].write(json.dumps(record, ensure_ascii=False) + "\n")
            handles["path_f"].flush()
            seen_path.add(ex_id)

        if ex_id not in seen_rankings:
            record = {"id": ex_id, "seq": input_output_seq, "rankings": ranking_example}
            handles["rankings_f"].write(json.dumps(record, ensure_ascii=False) + "\n")
            handles["rankings_f"].flush()
            seen_rankings.add(ex_id)

    for key in ("contri_f", "rankings_f", "path_f"):
        try:
            handles[key].close()
        except Exception:
            pass

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
