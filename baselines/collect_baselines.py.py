"""Merge all baseline scores into a single aligned JSON file."""
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KEY_ORDER = [
    "y_true", "hem_score",
    "p_true_score", "ppl_score", "RE_score", "SE_score",
    "embed_score", "utility_score",
    "att_score", "focus_score",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(values: list[Any], name: str) -> list[Any]:
    flat = []
    for i, v in enumerate(values):
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError(f"{name}[{i}] has length {len(v)}, expected scalar or [scalar].")
            flat.append(v[0])
        else:
            flat.append(v)
    return flat


def _check_len(name: str, values: list[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name}: got {len(values)}, expected {expected}.")


def _load_core_scores(path: Path) -> tuple[dict, dict]:
    """Load uncertainty_scores.json and extract core metrics."""
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object.")

    required = ["y_true", "hem_scores", "p_true_scores", "ppl_scores",
                "regular_entropy_scores", "semantic_entropy_scores"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    scores = {
        "y_true":     _flatten(payload["y_true"], "y_true"),
        "hem_score":  _flatten(payload["hem_scores"], "hem_scores"),
        "p_true_score": [math.exp(s) for s in _flatten(payload["p_true_scores"], "p_true_scores")],
        "ppl_score":  _flatten(payload["ppl_scores"], "ppl_scores"),
        "RE_score":   _flatten(payload["regular_entropy_scores"], "regular_entropy_scores"),
        "SE_score":   _flatten(payload["semantic_entropy_scores"], "semantic_entropy_scores"),
    }
    return payload, scores


def _validate_pt(path: Path, scores: dict) -> None:
    """Check hem/y_true alignment between pt file and uncertainty scores."""
    pt = torch.load(path, map_location="cpu", weights_only=False)
    if "hem" not in pt:
        raise KeyError(f"{path} missing key: hem")

    pt_hem = [float(x) for x in pt["hem"]]
    pt_y_true = [0 if x < 0.5 else 1 for x in pt_hem]

    # Use tolerance for float comparison
    if not all(abs(a - b) < 1e-6 for a, b in zip(pt_hem, scores["hem_score"])):
        raise ValueError(f"{path} hem does not align with uncertainty scores.")
    if pt_y_true != scores["y_true"]:
        raise ValueError(f"{path} y_true does not align with uncertainty scores.")


def _load_scores(path: Path | None, fallback: dict, key: str, entry_key: str) -> list[Any] | None:
    """Load optional scores from a JSON file or fallback dict."""
    if path is None:
        return fallback.get(key) if isinstance(fallback.get(key), list) else None

    payload = _load_json(path)

    if isinstance(payload, dict):
        return _flatten(payload[key], key)
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(x, dict) for x in payload):
            return [x.get(entry_key) for x in payload]
        return _flatten(payload, key)

    raise ValueError(f"Unsupported JSON shape in {path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--uncertainty_scores", default=None)
    parser.add_argument("--data_pt", default=None)
    parser.add_argument("--attention_results", default=None)
    parser.add_argument("--focus_results", default=None)
    parser.add_argument("--embed_scores", default=None)
    parser.add_argument("--utility_scores", default=None)
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    # Build path root from tags, or require explicit paths
    if args.model_tag and args.dataset and args.split:
        base = REPO_ROOT / "results" / f"{args.model_tag}_{args.dataset}"
        temp = base / "baselines" / "temp" / args.split
    else:
        base = None
        temp = None

    def _path(arg_val, default_rel):
        if arg_val:
            return Path(arg_val)
        if temp and default_rel:
            return temp / default_rel
        return None

    uncertainty_path = _path(args.uncertainty_scores, "other/uncertainty_scores.json")
    data_pt_path     = _path(args.data_pt, base / f"{args.dataset}_{args.split}_data.pt" if base else None)
    attention_path   = _path(args.attention_results, "attention/results.json")
    focus_path       = _path(args.focus_results, "focus/results.json")
    embed_path       = _path(args.embed_scores, "embedding/embed_score.json")
    utility_path     = _path(args.utility_scores, "utility_ranker/utility_score.json")
    out_path         = _path(args.out_path, base / "baselines" / f"baselines_{args.split}.json" if base else None)

    if out_path is None:
        raise ValueError("Provide --out_path or --model_tag/--dataset/--split.")

    # Core: uncertainty_scores.json is required
    if uncertainty_path is None:
        raise ValueError("Provide --uncertainty_scores or --model_tag/--dataset/--split.")
    if not uncertainty_path.exists():
        raise FileNotFoundError(f"Not found: {uncertainty_path}")

    fallback, core = _load_core_scores(uncertainty_path)
    expected = len(core["y_true"])

    # Validate alignment
    if data_pt_path and data_pt_path.exists():
        _validate_pt(data_pt_path, core)

    for k in ("y_true", "hem_score", "p_true_score", "ppl_score", "RE_score", "SE_score"):
        _check_len(k, core[k], expected)

    merged = {k: core[k] for k in ["y_true", "hem_score", "p_true_score", "ppl_score", "RE_score", "SE_score"]}

    # Optional baselines — files that exist get loaded, missing ones are skipped
    for label, path, json_key, entry_key in [
        ("attention", attention_path, "att_score", "AttentionScore"),
        ("focus",     focus_path,     "focus_score", "Focus"),
        ("embed",     embed_path,     "embed_score", "embed_score"),
        ("utility",   utility_path,   "utility_score", "utility_score"),
    ]:
        if path and path.exists():
            scores = _load_scores(path, fallback, json_key, entry_key)
            if scores is not None:
                _check_len(label, scores, expected)
                merged[json_key] = scores
                print(f"  + {label}: {path}")
        else:
            print(f"  - {label}: skipped (not found)")

    # Order by KEY_ORDER
    ordered = {k: merged[k] for k in KEY_ORDER if k in merged}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}  ({len(ordered)} metrics × {expected} samples)")


if __name__ == "__main__":
    main()
