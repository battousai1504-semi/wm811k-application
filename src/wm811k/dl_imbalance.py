from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_class_weights(
    labels: pd.Series,
    beta: float = 0.9999,
) -> pd.DataFrame:
    counts = labels.value_counts().sort_index()
    total_samples = int(counts.sum())
    num_classes = int(len(counts))

    balanced = total_samples / (num_classes * counts.astype(float))
    effective_number = 1.0 - np.power(beta, counts.astype(float))
    effective = (1.0 - beta) / effective_number
    effective = effective / effective.mean()

    weights = pd.DataFrame(
        {
            "label": counts.index.astype(str),
            "sample_count": counts.to_numpy(dtype=int),
            "class_frequency": counts.to_numpy(dtype=float) / total_samples,
            "balanced_weight": balanced.to_numpy(dtype=float),
            "effective_number_weight": effective.to_numpy(dtype=float),
        }
    )
    weights["balanced_weight_normalized"] = (
        weights["balanced_weight"] / weights["balanced_weight"].mean()
    )
    return weights


def add_sample_weights(
    metadata: pd.DataFrame,
    class_weights: pd.DataFrame,
    weight_column: str,
) -> pd.DataFrame:
    if weight_column not in class_weights.columns:
        raise KeyError(f"Unknown weight column: {weight_column}")

    weight_map = class_weights.set_index("label")[weight_column].to_dict()
    weighted = metadata.copy()
    weighted["sample_weight"] = weighted["label"].map(weight_map)

    if weighted["sample_weight"].isna().any():
        missing = weighted.loc[weighted["sample_weight"].isna(), "label"].unique()
        raise ValueError(f"Missing class weights for labels: {missing}")

    return weighted


def describe_imbalance(metadata: pd.DataFrame) -> dict[str, float | int | str]:
    counts = metadata["label"].value_counts()
    minority_label = str(counts.idxmin())
    majority_label = str(counts.idxmax())
    minority_count = int(counts.min())
    majority_count = int(counts.max())

    return {
        "total_samples": int(counts.sum()),
        "num_classes": int(len(counts)),
        "majority_label": majority_label,
        "majority_count": majority_count,
        "minority_label": minority_label,
        "minority_count": minority_count,
        "imbalance_ratio": float(majority_count / minority_count),
    }


def save_imbalance_artifacts(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    output_dir: Path,
    beta: float = 0.9999,
    sampler_weight_column: str = "effective_number_weight",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_weights = calculate_class_weights(train_metadata["label"], beta=beta)
    weighted_train_metadata = add_sample_weights(
        train_metadata,
        class_weights,
        weight_column=sampler_weight_column,
    )

    paths = {
        "class_weights": output_dir / "class_weights.csv",
        "train_sample_weights": output_dir / "train_sample_weights.csv",
        "split_class_counts": output_dir / "split_class_counts.csv",
        "config": output_dir / "imbalance_config.json",
        "summary": output_dir / "imbalance_summary.txt",
    }

    class_weights.to_csv(paths["class_weights"], index=False)
    weighted_train_metadata[
        ["wafer_id", "label", "label_encoded", "relative_path", "sample_weight"]
    ].to_csv(paths["train_sample_weights"], index=False)

    all_metadata = pd.concat(
        [
            train_metadata.assign(split="train"),
            validation_metadata.assign(split="validation"),
            test_metadata.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_counts = pd.crosstab(all_metadata["split"], all_metadata["label"])
    split_counts.to_csv(paths["split_class_counts"])

    train_description = describe_imbalance(train_metadata)
    config = {
        "beta": beta,
        "sampler_weight_column": sampler_weight_column,
        "recommended_loss": "FocalLoss(gamma=2.0, alpha=effective_number_weight)",
        "recommended_sampler": "WeightedRandomSampler using train_sample_weights.csv",
        "train_imbalance": train_description,
    }
    paths["config"].write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "DEEP LEARNING CLASS IMBALANCE SUMMARY",
        "=" * 44,
        f"Train samples: {train_description['total_samples']:,}",
        f"Classes: {train_description['num_classes']}",
        (
            "Majority class: "
            f"{train_description['majority_label']} "
            f"({train_description['majority_count']:,})"
        ),
        (
            "Minority class: "
            f"{train_description['minority_label']} "
            f"({train_description['minority_count']:,})"
        ),
        f"Imbalance ratio: {train_description['imbalance_ratio']:.2f}:1",
        "",
        "Recommended training setup:",
        "- Use WeightedRandomSampler from train_sample_weights.csv for the train loader.",
        "- Use FocalLoss with gamma=2.0 and effective-number class weights.",
        "- Keep validation/test distributions unchanged for honest evaluation.",
        "",
        "Class weights:",
        class_weights.round(6).to_string(index=False),
    ]
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")

    return paths
