#!/usr/bin/env python
"""
Run Focus uncertainty estimation on a prepared QA dataset.
"""
import argparse
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import Focus

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_idf(tokenizer_path, dataset, output_path):
    print(f"   Computing IDF from {len(dataset)} documents...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    doc_freq = defaultdict(int)

    for sample in tqdm(dataset, desc="Computing IDF"):
        text = sample.get("context", "")
        if not text:
            continue
        for token in set(tokenizer(text, add_special_tokens=False)["input_ids"]):
            doc_freq[token] += 1

    n_docs = len(dataset)
    idf = [math.log(n_docs / (doc_freq.get(i, -1) + 1)) for i in range(len(tokenizer.vocab))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(np.array(idf), f)
    print(f"   ✓ IDF saved to {output_path}")


def get_focus_estimator(model_path, dataset, idf_path):
    if not idf_path.exists():
        print("   Computing IDF (first run)...")
        compute_idf(model_path, dataset, idf_path)
    else:
        print(f"   ✓ Using cached IDF: {idf_path}")

    if not spacy.util.is_package("en_core_web_sm"):
        print("   Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")

    return Focus(
        model_name=model_path,
        path=str(idf_path),
        gamma=0.9,
        p=0.01,
        idf_dataset="",
        trust_remote_code=False,
        idf_seed=42,
        idf_dataset_size=-1,
        spacy_path="en_core_web_sm",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()

    data_path = REPO_ROOT / "results" / f"{args.model_tag}_{args.dataset}" / f"{args.dataset}_{args.split}_data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_path = REPO_ROOT / "results" / f"{args.model_tag}_{args.dataset}" / "baselines" / "temp" / args.split / "focus"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LM-Polygraph Focus Benchmark")
    print("=" * 80)
    print(f"Model Tag:  {args.model_tag}")
    print(f"Model Path: {args.model_path}")
    print(f"Dataset:    {args.dataset}")
    print(f"Split:      {args.split}")
    print(f"Data:       {data_path}")
    print(f"Output:     {output_path}")
    print("=" * 80)

    # ---- Load model ----
    print("\n📦 Loading model...")
    torch.set_grad_enabled(False)
    device_map = f"cuda:{args.gpu_id}"
    print(f"   Using device: {device_map}")

    model_kwargs = {
        "device_map": device_map,
        "attn_implementation": "eager",  # required for output_attentions
    }
    if "gemma" in args.model_path.lower():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("   Using bfloat16 for Gemma")
    else:
        model_kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    is_instruct = "llama" in args.model_path.lower() and "instruct" in args.model_path.lower()

    model = WhiteboxModel(
        base_model, tokenizer,
        model_path=args.model_path,
        generation_parameters=GenerationParameters(max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0),
        instruct=is_instruct,
    )
    print(f"✅ Model loaded! (instruct mode: {is_instruct})\n")

    # ---- Load data ----
    print(f"📂 Loading data from {data_path}...")
    pt_data = torch.load(str(data_path), weights_only=False)
    dataset = pt_data["data"]
    predictions = pt_data.get("pred", [])
    print(f"✅ Loaded {len(dataset)} samples\n")

    estimator = get_focus_estimator(args.model_path, dataset, output_path / "token_idf.pkl")

    print("🎯 Running Focus uncertainty estimation\n")

    results = []
    estimations_dir = output_path / "estimations"
    estimations_dir.mkdir(exist_ok=True)
    focus_scores = []

    print("🔄 Processing samples...")
    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        prompt = f"Answer the question in no more than five words. Context: {example['context']} Question: {example['question']} Answer:"

        ground_truth = ""
        if "answers" in example and isinstance(example["answers"], list) and len(example["answers"]) > 0:
            if isinstance(example["answers"][0], dict) and "text" in example["answers"][0]:
                ground_truth = example["answers"][0]["text"]

        result_entry = {
            "id": example.get("id", i),
            "question": example["question"],
            "context": example["context"][:200] + "..." if len(example["context"]) > 200 else example["context"],
            "ground_truth": ground_truth,
        }

        if predictions and i < len(predictions):
            result_entry["previous_predicted_answer"] = predictions[i].get("predicted_answer", "")

        try:
            ue_result = estimate_uncertainty(model, estimator, input_text=prompt)
            if "generated_answer" not in result_entry:
                result_entry["generated_answer"] = ue_result.generation_text
            ue_value = float(ue_result.uncertainty) if isinstance(ue_result.uncertainty, (int, float)) else float(ue_result.uncertainty.mean())
            if np.isnan(ue_value) or np.isinf(ue_value):
                print(f"\n⚠️  Warning: Focus returned {ue_value} for sample {i}")
                result_entry["Focus"] = None
                focus_scores.append(None)
            else:
                result_entry["Focus"] = ue_value
                focus_scores.append(ue_value)
        except Exception as e:
            print(f"\n⚠️  Error on sample {i}: {str(e)[:100]}")
            result_entry["Focus"] = None
            focus_scores.append(None)

        results.append(result_entry)

    # ---- Save ----
    print(f"\n💾 Saving results to {output_path}...")

    with open(output_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(results)
    df.to_csv(output_path / "results.csv", index=False)

    with open(estimations_dir / "Focus.json", "w") as f:
        json.dump(focus_scores, f, indent=2)

    config = {
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "data_path": str(data_path),
        "num_samples_processed": len(results),
        "methods": ["Focus"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    valid = [v for v in focus_scores if v is not None]
    if valid:
        print(f"{'Focus':40s}: mean={np.mean(valid):8.4f}, std={np.std(valid):8.4f}, min={np.min(valid):8.4f}, max={np.max(valid):8.4f}")

    print(f"\n✅ Done — {len(results)} samples → {output_path}")


if __name__ == "__main__":
    main()
