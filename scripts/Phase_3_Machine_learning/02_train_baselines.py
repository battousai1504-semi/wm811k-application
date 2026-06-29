import argparse
from pathlib import Path
import sys
import time

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.evaluation import evaluate_classifier, load_phase3_splits, validate_datasets
from wm811k.models import create_baseline_models
from wm811k.paths import MODEL_DIR, RESULT_DIR, SPLIT_DIR, ensure_dir


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.model_dir)
    ensure_dir(args.result_dir)

    print("=" * 70)
    print("PHASE 3.2 - BASELINE MODEL TRAINING")
    print("=" * 70)

    X_train, X_validation, y_train, y_validation = load_phase3_splits(args.split_dir)
    validate_datasets(X_train, X_validation, y_train, y_validation)

    print("\nDataset shapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"y_train:      {y_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"y_validation: {y_validation.shape}")

    label_encoder_path = args.model_dir / "label_encoder.joblib"
    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {label_encoder_path}\n"
            "Run 01_prepare_ml_data.py first."
        )

    label_encoder = joblib.load(label_encoder_path)
    print("\nClass mapping:")
    for encoded_label, class_name in enumerate(label_encoder.classes_):
        print(f"  {encoded_label} -> {class_name}")

    metrics_rows = []
    for model_name, model in create_baseline_models(args.random_state).items():
        print("\n" + "-" * 70)
        print(f"Training model: {model_name}")
        print("-" * 70)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        model_path = args.model_dir / f"{model_name}_baseline.joblib"
        joblib.dump(model, model_path)

        print(f"Training completed in {training_time:.4f} seconds.")
        print(f"Model saved to: {model_path}")

        metrics = evaluate_classifier(
            model=model,
            model_name=model_name,
            X_validation=X_validation,
            y_validation=y_validation,
            label_encoder=label_encoder,
            result_dir=args.result_dir,
        )
        metrics["training_time_seconds"] = training_time
        metrics_rows.append(metrics)

    comparison_dataframe = (
        pd.DataFrame(metrics_rows)
        .sort_values(by="macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    comparison_path = args.result_dir / "baseline_model_comparison.csv"
    comparison_dataframe.to_csv(comparison_path, index=False)

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_time_seconds",
    ]

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)
    print(comparison_dataframe[display_columns].round(4).to_string(index=False))

    best_row = comparison_dataframe.iloc[0]
    print("\nBest baseline model:")
    print(f"  Model:    {best_row['model']}")
    print(f"  Macro-F1: {best_row['macro_f1']:.4f}")
    print(f"\nComparison results saved to: {comparison_path}")
    print("Phase 3.2 completed successfully.")


if __name__ == "__main__":
    main()

