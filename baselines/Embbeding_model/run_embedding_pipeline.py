from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.Embbeding_model.llama import AttrConfig, LlamaAttributor


def load_pt_split(path: str) -> tuple[list[dict], list[int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload["data"]
    hem = payload["hem"].tolist()
    y_true = [0 if value < 0.5 else 1 for value in hem]
    return data, y_true


def build_prompt_dataset(data: list[dict], labels: list[int]) -> list[dict]:
    dataset = []
    for item, label in zip(data, labels):
        prompt = "Answer the question in no more than five words. "
        qc = f"{prompt}Context: {item['context']} Question: {item['question']} Answer:"
        dataset.append({"qc": qc, "label": label})
    return dataset


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def extract_hidden_features(
    tool: LlamaAttributor,
    prompt_dataset: list[dict],
    split_name: str,
) -> list[dict]:
    feature_rows: list[dict] = []
    for idx, item in enumerate(prompt_dataset):
        if idx % 100 == 0:
            print(f"[{split_name}] extracting hidden feature {idx}/{len(prompt_dataset)}")
        hidden = tool.get_last_hidden(item["qc"])
        hidden_mean = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
        feature_rows.append(
            {
                "qc": item["qc"],
                "label": item["label"],
                "hidden": hidden_mean.tolist(),
            }
        )
    return feature_rows


def feature_matrix(feature_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([row["hidden"] for row in feature_rows], dtype=np.float32)
    y = np.array([row["label"] for row in feature_rows], dtype=np.int32)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset_tag", required=True, choices=["squad2", "hotpot", "msmarco"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    model_tag = args.model_tag
    model_path = args.model_path
    dataset = args.dataset_tag
    dtype = args.dtype or ("bf16" if "gemma" in model_tag.lower() else "fp16")

    base = REPO_ROOT / "results"
    embed_dir = base / f"{model_tag}_{dataset}" / "baselines" / "temp" / "embedding"
    embed_dir.mkdir(parents=True, exist_ok=True)

    cfg = AttrConfig(
        model_path=model_path,
        device=args.device,
        dtype=dtype,
        metric="manhattan",
        i_block=10,
    )
    tool = LlamaAttributor(cfg)

    features_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        data_path = base / f"{model_tag}_{dataset}" / f"{dataset}_{split}_data.pt"
        split_dir = base / f"{model_tag}_{dataset}" / "baselines" / "temp" / split / "embedding"
        data, labels = load_pt_split(data_path)
        prompt_dataset = build_prompt_dataset(data, labels)
        dump_json(split_dir / "dataset.json", prompt_dataset)
        feature_rows = extract_hidden_features(tool, prompt_dataset, split)
        dump_json(split_dir / "hidden_features.json", feature_rows)
        features_by_split[split] = feature_rows

    x_train, y_train = feature_matrix(features_by_split["train"])
    x_val, y_val = feature_matrix(features_by_split["val"])
    x_test, y_test = feature_matrix(features_by_split["test"])

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=25,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        eval_metric=["auc", "logloss"],
    )
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=True)

    with (embed_dir / "xgboost_calibration.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    for split, x, y in (
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ):
        pred_probs = model.predict_proba(x)[:, 1]
        preds = model.predict(x)
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, pred_probs)
        split_dir = base / f"{model_tag}_{dataset}" / "baselines" / "temp" / split / "embedding"
        dump_json(split_dir / "embed_score.json", {"embed_score": pred_probs.tolist()})
        dump_json(
            split_dir / "metrics.json",
            {"accuracy": float(acc), "auc": float(auc), "num_samples": int(len(pred_probs))},
        )
        print(f"[{split}] accuracy={acc:.4f} auc={auc:.4f} saved={split_dir / 'embed_score.json'}")


if __name__ == "__main__":
    main()
