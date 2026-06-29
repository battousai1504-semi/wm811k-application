from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from wm811k.paths import ensure_dir
from wm811k.plots import save_confusion_matrix_image


def validate_datasets(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> None:
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different sample counts.")
    if len(X_validation) != len(y_validation):
        raise ValueError("X_validation and y_validation have different sample counts.")
    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError("Training and validation feature columns do not match.")
    if X_train.isnull().any().any() or X_validation.isnull().any().any():
        raise ValueError("Feature data contains missing values.")
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_validation.to_numpy()).all():
        raise ValueError("Feature data contains infinite values.")


def load_phase3_splits(split_dir):
    required_files = [
        split_dir / "X_train.csv",
        split_dir / "X_validation.csv",
        split_dir / "y_train.csv",
        split_dir / "y_validation.csv",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing split files:\n{missing_text}\nRun 01_prepare_ml_data.py first.")

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_validation = pd.read_csv(split_dir / "X_validation.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")["label"].to_numpy()
    y_validation = pd.read_csv(split_dir / "y_validation.csv")["label"].to_numpy()
    return X_train, X_validation, y_train, y_validation


def evaluate_classifier(
    model,
    model_name: str,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    label_encoder,
    result_dir,
) -> dict[str, float | str]:
    y_prediction = model.predict(X_validation)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_validation,
        y_prediction,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_validation,
        y_prediction,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_validation, y_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_validation, y_prediction),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    class_names = list(label_encoder.classes_)
    class_labels = np.arange(len(class_names))
    report = classification_report(
        y_validation,
        y_prediction,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_dataframe = pd.DataFrame(report).transpose()
    raw_matrix = confusion_matrix(y_validation, y_prediction, labels=class_labels)
    normalized_matrix = confusion_matrix(
        y_validation,
        y_prediction,
        labels=class_labels,
        normalize="true",
    )

    save_evaluation_results(
        model_name=model_name,
        metrics=metrics,
        report_dataframe=report_dataframe,
        raw_matrix=raw_matrix,
        normalized_matrix=normalized_matrix,
        y_true=y_validation,
        y_prediction=y_prediction,
        class_names=class_names,
        label_encoder=label_encoder,
        result_dir=result_dir,
    )
    print_evaluation_results(model_name, metrics, report_dataframe)
    return metrics


def save_evaluation_results(
    model_name: str,
    metrics: dict,
    report_dataframe: pd.DataFrame,
    raw_matrix: np.ndarray,
    normalized_matrix: np.ndarray,
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    class_names: list[str],
    label_encoder,
    result_dir,
) -> None:
    model_result_dir = ensure_dir(result_dir / model_name)

    pd.DataFrame([metrics]).to_csv(model_result_dir / "metrics.csv", index=False)
    report_dataframe.to_csv(model_result_dir / "classification_report.csv")

    pd.DataFrame(
        {
            "true_label_encoded": y_true,
            "predicted_label_encoded": y_prediction,
            "true_label": label_encoder.inverse_transform(y_true),
            "predicted_label": label_encoder.inverse_transform(y_prediction),
            "correct_prediction": y_true == y_prediction,
        }
    ).to_csv(model_result_dir / "validation_predictions.csv", index=False)

    raw_dataframe = pd.DataFrame(raw_matrix, index=class_names, columns=class_names)
    raw_dataframe.index.name = "Actual"
    raw_dataframe.columns.name = "Predicted"
    raw_dataframe.to_csv(model_result_dir / "confusion_matrix_raw.csv")

    normalized_dataframe = pd.DataFrame(normalized_matrix, index=class_names, columns=class_names)
    normalized_dataframe.index.name = "Actual"
    normalized_dataframe.columns.name = "Predicted"
    normalized_dataframe.to_csv(model_result_dir / "confusion_matrix_normalized.csv")

    save_confusion_matrix_image(
        raw_matrix,
        class_names,
        f"{model_name} - Raw Confusion Matrix",
        model_result_dir / "confusion_matrix_raw.png",
        "d",
    )
    save_confusion_matrix_image(
        normalized_matrix,
        class_names,
        f"{model_name} - Normalized Confusion Matrix",
        model_result_dir / "confusion_matrix_normalized.png",
        ".2f",
    )


def print_evaluation_results(
    model_name: str,
    metrics: dict,
    report_dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ):
        print(f"{key:20s}: {metrics[key]:.4f}")

    class_rows = report_dataframe.loc[
        ~report_dataframe.index.isin(["accuracy", "macro avg", "weighted avg"])
    ]
    print("\nPer-class performance:")
    print(class_rows[["precision", "recall", "f1-score", "support"]].round(4))

