import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.features import DEFAULT_FEATURE_COLUMNS, TEXTURE_FEATURE_COLUMNS
from wm811k.ml_data import find_label_column
from wm811k.paths import FEATURE_DATA_PATH, MODEL_DIR, RESULT_DIR, ensure_dir
from wm811k.plots import save_confusion_matrix_image


DEFAULT_SELECTED_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
    / "feature_analysis"
    / "05_selected_features.csv"
)
DEFAULT_RESULT_DIR = RESULT_DIR.parent / "classical_svm"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=FEATURE_DATA_PATH)
    parser.add_argument("--selected-features", type=Path, default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--feature-set",
        choices=["selected_plus_texture", "default_plus_texture", "texture_only"],
        default="selected_plus_texture",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help=(
            "Maximum samples per class for SVM RBF 5-fold CV. "
            "Use 0 to run on every available labeled sample."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--cache-mb", type=float, default=1000.0)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_selected_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            "Selected-feature file not found. Run 05_feature_analysis.py first:\n"
            f"{path}"
        )

    dataframe = pd.read_csv(path)
    if "selected_feature" not in dataframe.columns:
        raise ValueError(f"Missing 'selected_feature' column in: {path}")

    features = dataframe["selected_feature"].dropna().astype(str).tolist()
    if not features:
        raise ValueError(f"No selected features were found in: {path}")

    return features


def choose_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.feature_set == "selected_plus_texture":
        base_features = load_selected_feature_columns(args.selected_features)
    elif args.feature_set == "default_plus_texture":
        base_features = list(DEFAULT_FEATURE_COLUMNS)
    else:
        base_features = []

    return base_features + list(TEXTURE_FEATURE_COLUMNS)


def validate_feature_columns(dataframe: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in dataframe.columns]
    non_numeric = [
        column
        for column in feature_columns
        if column in dataframe.columns
        and not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if missing or non_numeric:
        raise ValueError(
            "The feature CSV is not ready for the classical SVM baseline.\n"
            f"Missing columns: {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}\n"
            f"Non-numeric columns: {non_numeric}\n"
            "Run 02_batch_features.py again so the CSV includes LBP/GLCM features."
        )


def stratified_limit(
    dataframe: pd.DataFrame,
    label_column: str,
    max_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    if max_per_class <= 0:
        return dataframe.copy()

    parts = []
    for _, group in dataframe.groupby(label_column, sort=False):
        n_samples = min(len(group), max_per_class)
        parts.append(group.sample(n=n_samples, random_state=random_state))

    return pd.concat(parts, ignore_index=True)


def clean_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    clean = dataframe[["wafer_id", label_column] + feature_columns].copy()
    clean[feature_columns] = clean[feature_columns].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=feature_columns + [label_column])
    return clean


def make_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=args.c,
                    gamma=args.gamma,
                    class_weight="balanced",
                    cache_size=args.cache_mb,
                    decision_function_shape="ovr",
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_prediction,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_prediction,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_prediction),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def run_cross_validation(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    wafer_ids: pd.Series,
    label_encoder: LabelEncoder,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )

    fold_rows = []
    prediction_frames = []
    out_of_fold_prediction = np.empty_like(y)

    for fold_index, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        print("\n" + "-" * 70)
        print(f"SVM RBF fold {fold_index}/{args.folds}")
        print("-" * 70)

        fold_model = clone(model)
        start_time = time.perf_counter()
        fold_model.fit(X.iloc[train_index], y[train_index])
        training_time = time.perf_counter() - start_time

        y_prediction = fold_model.predict(X.iloc[validation_index])
        out_of_fold_prediction[validation_index] = y_prediction

        metrics = calculate_metrics(y[validation_index], y_prediction)
        metrics["fold"] = fold_index
        metrics["train_samples"] = len(train_index)
        metrics["validation_samples"] = len(validation_index)
        metrics["training_time_seconds"] = training_time
        fold_rows.append(metrics)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_index,
                    "wafer_id": wafer_ids.iloc[validation_index].to_numpy(),
                    "true_label_encoded": y[validation_index],
                    "predicted_label_encoded": y_prediction,
                    "true_label": label_encoder.inverse_transform(y[validation_index]),
                    "predicted_label": label_encoder.inverse_transform(y_prediction),
                    "correct_prediction": y[validation_index] == y_prediction,
                }
            )
        )

        print(
            "Fold metrics: "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"time={training_time:.2f}s"
        )

    return (
        pd.DataFrame(fold_rows),
        pd.concat(prediction_frames, ignore_index=True),
        out_of_fold_prediction,
    )


def save_results(
    result_dir: Path,
    model: Pipeline,
    feature_columns: list[str],
    label_encoder: LabelEncoder,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    y: np.ndarray,
    out_of_fold_prediction: np.ndarray,
    args: argparse.Namespace,
) -> None:
    ensure_dir(result_dir)

    fold_metrics.to_csv(result_dir / "svm_rbf_5fold_metrics.csv", index=False)
    predictions.to_csv(result_dir / "svm_rbf_5fold_predictions.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(
        result_dir / "svm_rbf_feature_columns.csv",
        index=False,
    )

    class_names = list(label_encoder.classes_)
    class_labels = np.arange(len(class_names))
    report = classification_report(
        y,
        out_of_fold_prediction,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_dataframe = pd.DataFrame(report).transpose()
    report_dataframe.to_csv(result_dir / "svm_rbf_5fold_classification_report.csv")

    raw_matrix = confusion_matrix(y, out_of_fold_prediction, labels=class_labels)
    normalized_matrix = confusion_matrix(
        y,
        out_of_fold_prediction,
        labels=class_labels,
        normalize="true",
    )
    pd.DataFrame(raw_matrix, index=class_names, columns=class_names).to_csv(
        result_dir / "svm_rbf_5fold_confusion_matrix_raw.csv"
    )
    pd.DataFrame(normalized_matrix, index=class_names, columns=class_names).to_csv(
        result_dir / "svm_rbf_5fold_confusion_matrix_normalized.csv"
    )

    save_confusion_matrix_image(
        raw_matrix,
        class_names,
        "SVM RBF 5-fold - Raw Confusion Matrix",
        result_dir / "svm_rbf_5fold_confusion_matrix_raw.png",
        "d",
    )
    save_confusion_matrix_image(
        normalized_matrix,
        class_names,
        "SVM RBF 5-fold - Normalized Confusion Matrix",
        result_dir / "svm_rbf_5fold_confusion_matrix_normalized.png",
        ".2f",
    )

    summary_metrics = calculate_metrics(y, out_of_fold_prediction)
    mean_metrics = fold_metrics[
        [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "training_time_seconds",
        ]
    ].mean()
    std_metrics = fold_metrics[
        [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "training_time_seconds",
        ]
    ].std(ddof=0)

    summary_lines = [
        "CLASSICAL CV SVM RBF BASELINE SUMMARY",
        "=" * 44,
        f"Feature set: {args.feature_set}",
        f"Feature count: {len(feature_columns)}",
        f"Samples used: {len(y):,}",
        f"Max per class: {'all' if args.max_per_class <= 0 else args.max_per_class}",
        f"Folds: {args.folds}",
        f"SVM C: {args.c}",
        f"SVM gamma: {args.gamma}",
        "",
        "Out-of-fold metrics:",
        *[f"{key}: {value:.4f}" for key, value in summary_metrics.items()],
        "",
        "Fold mean +/- std:",
        *[
            f"{key}: {mean_metrics[key]:.4f} +/- {std_metrics[key]:.4f}"
            for key in mean_metrics.index
        ],
        "",
        "Outputs:",
        str(result_dir.resolve()),
    ]
    (result_dir / "svm_rbf_5fold_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    joblib.dump(model, result_dir / "svm_rbf_final_pipeline.joblib")
    joblib.dump(label_encoder, result_dir / "svm_rbf_label_encoder.joblib")


def main() -> None:
    args = parse_args()
    ensure_dir(args.result_dir)
    ensure_dir(args.model_dir)

    dataframe = pd.read_csv(args.csv)
    label_column = find_label_column(dataframe)
    feature_columns = choose_feature_columns(args)
    validate_feature_columns(dataframe, feature_columns)

    clean = clean_dataframe(dataframe, feature_columns, label_column)
    sampled = stratified_limit(
        clean,
        label_column=label_column,
        max_per_class=args.max_per_class,
        random_state=args.random_state,
    )

    print("=" * 70)
    print("PHASE 3.3 - CLASSICAL CV SVM RBF BASELINE")
    print("=" * 70)
    print(f"Feature CSV: {args.csv}")
    print(f"Feature set: {args.feature_set}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Rows before SVM sampling: {len(clean):,}")
    print(f"Rows used for SVM CV: {len(sampled):,}")
    print("\nClass distribution:")
    print(sampled[label_column].value_counts())

    X = sampled[feature_columns].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(sampled[label_column].astype(str))

    model = make_model(args)
    fold_metrics, predictions, out_of_fold_prediction = run_cross_validation(
        model=model,
        X=X,
        y=y,
        wafer_ids=sampled["wafer_id"],
        label_encoder=label_encoder,
        args=args,
    )

    print("\nTraining final SVM RBF pipeline on the SVM baseline dataset...")
    start_time = time.perf_counter()
    model.fit(X, y)
    final_training_time = time.perf_counter() - start_time
    print(f"Final SVM training completed in {final_training_time:.2f}s.")

    save_results(
        result_dir=args.result_dir,
        model=model,
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        fold_metrics=fold_metrics,
        predictions=predictions,
        y=y,
        out_of_fold_prediction=out_of_fold_prediction,
        args=args,
    )

    model_path = args.model_dir / "svm_rbf_classical_baseline.joblib"
    joblib.dump(model, model_path)

    comparison = fold_metrics.drop(
        columns=["fold", "train_samples", "validation_samples"],
        errors="ignore",
    ).agg(["mean", "std"])
    print("\nSVM RBF 5-fold metrics:")
    print(comparison.round(4).to_string())
    print(f"\nSaved final SVM model to: {model_path}")
    print(f"Saved SVM baseline outputs to: {args.result_dir.resolve()}")
    print("Phase 3.3 completed successfully.")


if __name__ == "__main__":
    main()
