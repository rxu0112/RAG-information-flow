import argparse
import os
import numpy as np
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility_Ranker.utils.utils import load_file, save_file_jsonl 


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def nliEval(model, tokenizer, premise_raw, hypothesis_raw):
    batch_tokens = tokenizer.batch_encode_plus(
        list(zip(premise_raw, hypothesis_raw)),
        padding=True,
        max_length=512,
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        model_outputs = model(**{k: v.to(model.device) for k, v in batch_tokens.items()})
    batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
    batch_evids = batch_probs[:, 0].tolist()
    return batch_evids


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nli_model',
        type=str,
        default=str(PROJECT_ROOT / "retrieval_qa" / "albert-xlarge-vitaminc-mnli"),
    )
    parser.add_argument('--ares_model', type=str, default=None)
    parser.add_argument(
        '--input_file',
        type=str,
        default="",
    )
    parser.add_argument(
        '--result_fp',
        type=str,
        default="",
    )
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    
    args = parser.parse_args()

    input_data = load_file(args.input_file)
    os.makedirs(Path(args.result_fp).parent, exist_ok=True)

    if args.nli_model is not None:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model, use_fast=False)
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nli_model.to(device)
        if device.type == "cuda":
            nli_model.half()

        for item in tqdm(input_data, desc="Processing items", unit="item"):
            premise = []
            hypothesis = []
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                premise.append(ctx["text"])
                hypothesis.append(item["question"] + " " + ctx.get("output", ""))

            entail_results = nliEval(nli_model, nli_tokenizer, premise, hypothesis)
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                ctx["NLI"] = entail_results[i]

    if args.ares_model is not None:
        print('Not implemented')
        exit(0)

    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


if __name__ == "__main__":
    main()
