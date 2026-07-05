# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import pickle
import time
import glob
from pathlib import Path
import sys

import numpy as np
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Utility_Ranker.contriever.src.index
import Utility_Ranker.contriever.src.contriever
import Utility_Ranker.contriever.src.utils
import Utility_Ranker.contriever.src.slurm
import Utility_Ranker.contriever.src.data
from Utility_Ranker.contriever.src.evaluation import calculate_matches
import Utility_Ranker.contriever.src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_input_paths(path_pattern, description):
    normalized_pattern = os.path.expanduser(path_pattern.strip()).strip('"').strip("'")
    input_paths = sorted(glob.glob(normalized_pattern))
    if not input_paths and os.path.exists(normalized_pattern):
        input_paths = [normalized_pattern]

    if input_paths:
        return input_paths

    parent_dir = os.path.dirname(normalized_pattern) or "."
    parent_hint = ""
    if os.path.isdir(parent_dir):
        sibling_paths = sorted(os.listdir(parent_dir))
        preview = sibling_paths[:10]
        suffix = " ..." if len(sibling_paths) > len(preview) else ""
        parent_hint = f" Available entries in {parent_dir}: {preview}{suffix}"

    raise FileNotFoundError(
        f"No {description} matched pattern {path_pattern!r} "
        f"(normalized to {normalized_pattern!r}).{parent_hint}"
    )


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = Utility_Ranker.contriever.src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[int(doc_id)] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                # "title": docs[c]["title"],
                "text": docs[c].get("context", docs[c].get("text")),
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)

    for example in data:
        answers = example.get("answers", example.get("answer", []))
        if answers is None:
            answers = []
        elif isinstance(answers, str):
            answers = [answers] if answers else []
        elif not isinstance(answers, list):
            answers = [answers]
        example["answers"] = answers

    return data


def main(args):

    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = Utility_Ranker.contriever.src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    index = Utility_Ranker.contriever.src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    # index all passages
    input_paths = resolve_input_paths(args.passages_embeddings, "passage embedding files")
    embeddings_dir = os.path.dirname(input_paths[0]) or "."
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    # passages = src.data.load_passages(args.passages)
    # passage_id_map = {x["id"]: x for x in passages}
    passage_id_map = Utility_Ranker.contriever.src.data.load_passages(args.passages)

    data_paths = resolve_input_paths(args.data, "data files")
    for path in data_paths:
        data = load_data(path)
        print(len(data))
        output_name = f"{Path(path).stem}_retrieval.jsonl"
        output_path = os.path.join(args.output_dir, output_name)

        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map, top_ids_and_scores)
        hasanswer = validate(data, args.validation_workers)
        add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--passages",
        type=str,
        default="",
        help="Path to passages (.jsonl or .tsv file)",
    )
    parser.add_argument(
        "--passages_embeddings",
        type=str,
        default="",
        help="Glob path to encoded passages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Results are written to output_dir with data suffix",
    )
    parser.add_argument("--n_docs", type=int, default=5, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=900, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=str(PROJECT_ROOT / "contriever_msmarco"),
        help="Path to directory containing model weights and config file",
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=10, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    Utility_Ranker.contriever.src.slurm.init_distributed_mode(args)
    main(args)
