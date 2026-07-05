from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

DATASET_CHOICES = ("squad2", "hotpot", "msmarco")
SPLIT_CHOICES = ("train", "val", "test")


@dataclass(frozen=True)
class BaselineLayout:
    model_tag: str
    dataset_tag: str
    split: str
    data_pt_path: Path
    merged_output_path: Path
    other_dir: Path
    attention_dir: Path
    focus_dir: Path
    utility_dir: Path
    embedding_dir: Path


def build_baseline_layout(model_tag: str, dataset_tag: str, split: str) -> BaselineLayout:
    run_dir = RESULTS_ROOT / f"{model_tag}_{dataset_tag}"
    baselines_dir = run_dir / "baselines"
    split_temp_dir = baselines_dir / "temp" / split

    return BaselineLayout(
        model_tag=model_tag,
        dataset_tag=dataset_tag,
        split=split,
        data_pt_path=run_dir / f"{dataset_tag}_{split}_data.pt",
        merged_output_path=baselines_dir / f"baselines_{split}.json",
        other_dir=split_temp_dir / "other",
        attention_dir=split_temp_dir / "attention",
        focus_dir=split_temp_dir / "focus",
        utility_dir=split_temp_dir / "utility_ranker",
        embedding_dir=split_temp_dir / "embedding",
    )
