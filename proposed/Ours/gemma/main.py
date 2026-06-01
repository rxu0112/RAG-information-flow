import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from Ours.gemma.gemma import GemmaAttributor, AttrConfig, limit_dataset_examples
from datasets import load_dataset
from tqdm import tqdm
import heapq
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_CHOICES = ("squad2", "hotpot", "msmarco")
DTYPE_NAME = "bf16"
METRIC_NAME = "manhattan"
MODEL_TAG = "gemma-3-4B-it"
DEFAULT_MODEL_PATH = "MODEL_PATH_PLACEHOLDER"


class OrderedSet:
    def __init__(self):
        self._data = {}

    def add(self, item):
        self._data[item] = None  # overwrite if exists (no duplicate)

    def __iter__(self):
        return iter(self._data.keys())

    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"OrderedSet({list(self._data.keys())})"


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_seen_ids(jsonl_path: str):
    seen = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        seen.add(obj["id"])
                except Exception:
                    pass
    return seen


def _data_path(dataset_name: str) -> Path:
    return REPO_ROOT / "preprocessed_data" / f"{dataset_name}_prepared.json"


def _output_dir(dataset_name: str) -> Path:
    return REPO_ROOT / "results" / f"{MODEL_TAG}_{dataset_name}" / f"loop_{DTYPE_NAME}"


def compute_ranking_from_layer_mats(device, matrices_arrays):
    assert len(matrices_arrays) > 0, "empty matrices_arrays"
    mats = []
    for m in matrices_arrays:
        if hasattr(m, "detach"):
            m = m.detach().float().cpu().numpy()
        m = np.asarray(m)
        assert m.ndim == 2 and m.shape[0] == m.shape[1], "each layer must be (T,T)"
        mats.append(m.astype(np.float32, copy=False))

    T = mats[0].shape[0]
    for m in mats:
        assert m.shape == (T, T), "all layer mats must be the same shape"

    recorded_nodes = OrderedSet()
    start = T - 1
    recorded_nodes.add(start)

    candidates = {str(start): [arr[start, :start].copy() for arr in mats]}

    extracted_matrices = [np.zeros((T, T), dtype=np.float32) for _ in range(len(mats))]
    for i in range(len(extracted_matrices)):
        extracted_matrices[i][-1, -1] = mats[i][-1, -1]

    heap = []
    for group, arrays in candidates.items():
        for j, arr in enumerate(arrays):  # arr shape=(start,)
            for k, val in enumerate(arr):
                heapq.heappush(heap, (-float(val), group, j, k))

    while len(recorded_nodes) < start + 1:
        if not heap:
            for j in range(start - 1, -1, -1):
                if j not in recorded_nodes:
                    recorded_nodes.add(j)
            break

        neg_val, group, arr_idx, pos_idx = heapq.heappop(heap)
        recorded_nodes.add(pos_idx)

        g = int(group)  
        extracted_matrices[arr_idx][g, pos_idx] = mats[arr_idx][g, pos_idx]
        for a, mat in enumerate(extracted_matrices[:arr_idx]):
            mat[pos_idx, pos_idx] = mats[a][pos_idx, pos_idx]
        key = str(pos_idx)
        if key not in candidates:
            candidates[key] = []

        old_len = len(candidates[key])
        if arr_idx > old_len:
            new_part = [arr[pos_idx, :pos_idx].copy() for arr in mats[old_len:arr_idx]]
            candidates[key].extend(new_part)

            for j, arr in enumerate(new_part, start=old_len):
                for k, val in enumerate(arr):
                    heapq.heappush(heap, (-float(val), key, j, k))

    ranking = list(recorded_nodes)

    P = torch.from_numpy(extracted_matrices[0]).to(device=device, dtype=torch.float32)
    for i in range(1, len(extracted_matrices)):
        mat = torch.from_numpy(extracted_matrices[i]).to(device=device, dtype=torch.float32)
        P = torch.matmul(mat, P)

    last_row_P = P[-1, :]
    last_p = [last_row_P.cpu().tolist()]

    return ranking, last_p


def loop_result(tool, device: str, prompt: str):

    init_enc = tool.tokenizer(prompt, return_tensors="pt")
    init_enc = {k: v.to(device) for k, v in init_enc.items()}
    init_input_ids = init_enc["input_ids"]  # [1, T]
    init_tokens = [tool.tokenizer.decode([tid], skip_special_tokens=False)
                   for tid in init_input_ids[0].tolist()]
    batch_size, seq_len_init = init_input_ids.shape

    contrib_per_step = []
    rankings_per_step = []
    path_per_step = []

    for _ in range(10):
        next_id, next_txt = tool.generate_one_token(
            prompt, do_sample=False, temperature=1.0, top_p=1.0
        )

        if tool.tokenizer.eos_token_id is not None and next_id == tool.tokenizer.eos_token_id:
            break
        if next_txt.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>"}:
            break

        enc = tool.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]  # [1, T]
        batch_size, seq_len = input_ids.shape
        attention_mask = enc['attention_mask']
        oproj_refs, outputs_after_mlp, outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs = tool._capture_o_proj_refs(enc)
        contrib_mats = tool._manual_forward_once_and_calc_matrices(input_ids, oproj_refs, outputs_after_mlp, outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs, attention_mask)
        contrib = tool._accumulate_last_token(device, contrib_mats, seq_len)

        prompt += next_txt
        contrib_per_step.append(contrib)

        mats_np = []
        for m in contrib_mats:
            if hasattr(m, "detach"):
                mats_np.append(m.detach().float().cpu().numpy())
            else:
                mats_np.append(np.asarray(m, dtype=np.float16))

        ranking, last_p = compute_ranking_from_layer_mats(device, mats_np)
        rankings_per_step.append(ranking)
        path_per_step.append(last_p)

    return prompt, contrib_per_step, rankings_per_step, path_per_step


def init_dist():
    local_division = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_division)

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    division = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    return division, local_division, world_size


def shard_indices_contiguous(n_items: int, world_size: int, rank: int):
    base = n_items // world_size
    rem = n_items % world_size

    if rank < rem:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base

    return list(range(start, min(end, n_items)))


def shard_indices(n_items, world_size, rank):
    return [i for i in range(n_items) if i % world_size == rank]


def main():
    parser = argparse.ArgumentParser(
        description="Compute last-layer token contributions using GemmaAttributor (distributed)."
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--i_block", type=int, default=100) # Number of rows processed per block when computing contribution matrices; smaller values reduce peak GPU memory usage but usually make the run slower.
    args = parser.parse_args()
    resolved_data_path = _data_path(args.dataset)
    output_dir = _output_dir(args.dataset)
    division, local_division, world_size = init_dist()
    device = f"cuda:{local_division}"
    torch.set_grad_enabled(False)

    if division == 0:
        print(
            f"[INFO] world_size={world_size} | data_path={resolved_data_path} "
            f"| model={MODEL_TAG} | model_path={args.model_path} "
            f"| dtype={DTYPE_NAME} | metric={METRIC_NAME}"
        )
    dataset = load_dataset("json", data_files={"validation": str(resolved_data_path)})["validation"]
    dataset = limit_dataset_examples(dataset, args.dataset)
    n_all = len(dataset)
    idxs = shard_indices_contiguous(n_all, world_size, division)
    shard = dataset.select(idxs)

    if division == 0:
        lens = [len(shard_indices_contiguous(n_all, world_size, r)) for r in range(world_size)]
        print(f"[INFO] dataset={n_all}, per-division counts={lens}")

    cfg = AttrConfig(
        model_path=args.model_path,
        device=device,
        i_block=args.i_block,
    )
    tool = GemmaAttributor(cfg)
    contri_jsonl = str(output_dir / f"loop_{METRIC_NAME}_contri_{DTYPE_NAME}_{division}.jsonl")
    rankings_jsonl = str(output_dir / f"loop_{METRIC_NAME}_rank_{DTYPE_NAME}_{division}.jsonl")
    path_jsonl = str(output_dir / f"loop_{METRIC_NAME}_path_{DTYPE_NAME}_{division}.jsonl")

    _ensure_dir(contri_jsonl)
    _ensure_dir(rankings_jsonl)
    _ensure_dir(path_jsonl)

    seen_contri = _load_seen_ids(contri_jsonl)
    seen_rankings = _load_seen_ids(rankings_jsonl)
    seen_path = _load_seen_ids(path_jsonl)

    handles = {
        "contri_f": open(contri_jsonl, "a", encoding="utf-8", buffering=1),
        "rankings_f": open(rankings_jsonl, "a", encoding="utf-8", buffering=1),
        "path_f": open(path_jsonl, "a", encoding="utf-8", buffering=1),
    }

    pbar = tqdm(range(len(shard)), desc=f"Division{division} (GPU {local_division})", disable=(division != 0))
    for i in pbar:
        example = shard[i]
        ex_id = example.get("id", None) or f"division{division}_idx{i}"
        context = example["context"]
        question = example["question"]
        prompt = tool.build_prompt(context, question)

        if ex_id in seen_contri and ex_id in seen_rankings and ex_id in seen_path:
            continue
        input_output_seq, contri_loop, ranking_example, path_steps = loop_result(
            tool=tool, device=device, prompt=prompt
        )

        if ex_id not in seen_contri:
            rec = {"id": ex_id, "seq": input_output_seq, "contri": contri_loop}
            handles["contri_f"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles["contri_f"].flush()
            seen_contri.add(ex_id)

        if ex_id not in seen_path:
            rec = {"id": ex_id, "seq": input_output_seq, "path": path_steps}
            handles["path_f"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles["path_f"].flush()
            seen_path.add(ex_id)

        if ex_id not in seen_rankings:
            rec = {"id": ex_id, "seq": input_output_seq, "rankings": ranking_example}
            handles["rankings_f"].write(json.dumps(rec, ensure_ascii=False) + "\n")
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
