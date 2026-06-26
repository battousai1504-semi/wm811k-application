from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3_splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classical_ml"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase3_ml"
    / "baselines"
)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_datasets():
    """
    Load training and validation datasets created in Phase 3.1.

    Returns
    -------
    X_train : pandas.DataFrame
        Standardized training features.

    X_validation : pandas.DataFrame
        Standardized validation features.

    y_train : numpy.ndarray
        Encoded training labels.

    y_validation : numpy.ndarray
        Encoded validation labels.
    """

    required_files = [
        SPLIT_DIR / "X_train.csv",
        SPLIT_DIR / "X_validation.csv",
        SPLIT_DIR / "y_train.csv",
        SPLIT_DIR / "y_validation.csv"
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}\n"
                "Please run 01_prepare_ml_data.py first."
            )

    X_train = pd.read_csv(
        SPLIT_DIR / "X_train.csv"
    )

    X_validation = pd.read_csv(
        SPLIT_DIR / "X_validation.csv"
    )

    y_train = pd.read_csv(
        SPLIT_DIR / "y_train.csv"
    )["label"].to_numpy()

    y_validation = pd.read_csv(
        SPLIT_DIR / "y_validation.csv"
    )["label"].to_numpy()

    return (
        X_train,
        X_validation,
        y_train,
        y_validation
    )


# ============================================================
# 3. DATA VALIDATION
# ============================================================

def validate_datasets(
    X_train,
    X_validation,
    y_train,
    y_validation
):
    """
    Check whether the loaded datasets are valid.
    """

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train have different sample counts."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation and y_validation "
            "have different sample counts."
        )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if X_train.isnull().any().any():
        raise ValueError(
            "X_train contains missing values."
        )

    if X_validation.isnull().any().any():
        raise ValueError(
            "X_validation contains missing values."
        )

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError(
            "X_train contains infinite values."
        )

    if not np.isfinite(X_validation.to_numpy()).all():
        raise ValueError(
            "X_validation contains infinite values."
        )

    print("\nDataset validation completed successfully.")


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    X_validation,
    y_validation,
    label_encoder
):
    """
    Evaluate a trained classifier on the validation set.

    The function calculates:
    - Accuracy
    - Balanced Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-score
    - Weighted F1-score
    - Classification Report
    - Confusion Matrix
    """

    y_prediction = model.predict(X_validation)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            y_prediction,
            average="macro",
            zero_division=0
        )
    )

    _, _, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            y_prediction,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        y_validation,
        y_prediction
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        y_prediction
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    class_names = label_encoder.classes_

    class_labels = np.arange(
        len(class_names)
    )

    report = classification_report(
        y_validation,
        y_prediction,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_dataframe = (
        pd.DataFrame(report)
        .transpose()
    )

    matrix = confusion_matrix(
        y_validation,
        y_prediction,
        labels=class_labels
    )

    normalized_matrix = confusion_matrix(
        y_validation,
        y_prediction,
        labels=class_labels,
        normalize="true"
    )

    save_evaluation_results(
        model_name=model_name,
        metrics=metrics,
        report_dataframe=report_dataframe,
        confusion_matrix_raw=matrix,
        confusion_matrix_normalized=normalized_matrix,
        y_true=y_validation,
        y_prediction=y_prediction,
        label_encoder=label_encoder
    )

    print_evaluation_results(
        model_name=model_name,
        metrics=metrics,
        report_dataframe=report_dataframe
    )

    return metrics


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_evaluation_results(
    model_name,
    metrics,
    report_dataframe,
    confusion_matrix_raw,
    confusion_matrix_normalized,
    y_true,
    y_prediction,
    label_encoder
):
    """
    Save metrics, classification report, predictions,
    and confusion matrices.
    """

    model_result_dir = RESULT_DIR / model_name

    model_result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    class_names = label_encoder.classes_

    # --------------------------------------------------------
    # Save metric summary
    # --------------------------------------------------------

    pd.DataFrame([metrics]).to_csv(
        model_result_dir / "metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_dataframe.to_csv(
        model_result_dir / "classification_report.csv"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_dataframe = pd.DataFrame({
        "true_label_encoded": y_true,
        "predicted_label_encoded": y_prediction,
        "true_label": label_encoder.inverse_transform(
            y_true
        ),
        "predicted_label": label_encoder.inverse_transform(
            y_prediction
        ),
        "correct_prediction": y_true == y_prediction
    })

    prediction_dataframe.to_csv(
        model_result_dir / "validation_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save raw confusion matrix as CSV
    # --------------------------------------------------------

    raw_matrix_dataframe = pd.DataFrame(
        confusion_matrix_raw,
        index=class_names,
        columns=class_names
    )

    raw_matrix_dataframe.index.name = "Actual"
    raw_matrix_dataframe.columns.name = "Predicted"

    raw_matrix_dataframe.to_csv(
        model_result_dir / "confusion_matrix_raw.csv"
    )

    # --------------------------------------------------------
    # Save normalized confusion matrix as CSV
    # --------------------------------------------------------

    normalized_matrix_dataframe = pd.DataFrame(
        confusion_matrix_normalized,
        index=class_names,
        columns=class_names
    )

    normalized_matrix_dataframe.index.name = "Actual"
    normalized_matrix_dataframe.columns.name = "Predicted"

    normalized_matrix_dataframe.to_csv(
        model_result_dir
        / "confusion_matrix_normalized.csv"
    )

    # --------------------------------------------------------
    # Save raw confusion matrix image
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_raw,
        display_labels=class_names
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    axis.set_title(
        f"{model_name} — Raw Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        model_result_dir
        / "confusion_matrix_raw.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    # --------------------------------------------------------
    # Save normalized confusion matrix image
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_normalized,
        display_labels=class_names
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format=".2f",
        colorbar=False
    )

    axis.set_title(
        f"{model_name} — Normalized Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        model_result_dir
        / "confusion_matrix_normalized.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

def print_evaluation_results(
    model_name,
    metrics,
    report_dataframe
):
    """
    Print the main validation results.
    """

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1-score:    "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1-score: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print("\nPer-class performance:")

    columns_to_display = [
        "precision",
        "recall",
        "f1-score",
        "support"
    ]

    class_rows = report_dataframe.loc[
        ~report_dataframe.index.isin([
            "accuracy",
            "macro avg",
            "weighted avg"
        ])
    ]

    print(
        class_rows[columns_to_display]
        .round(4)
    )


# ============================================================
# 7. CREATE BASELINE MODELS
# ============================================================

def create_models():
    """
    Create baseline machine learning models.

    No hyperparameter tuning is performed in Phase 3.2.
    """

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ),

        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():
    """
    Train and evaluate baseline machine learning models.
    """

    print("=" * 70)
    print("PHASE 3.2 — BASELINE MODEL TRAINING")
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = load_datasets()

    print("\nDataset shapes:")

    print(
        f"X_train:      {X_train.shape}"
    )

    print(
        f"y_train:      {y_train.shape}"
    )

    print(
        f"X_validation: {X_validation.shape}"
    )

    print(
        f"y_validation: {y_validation.shape}"
    )

    print(
        f"Number of features: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Validate datasets
    # --------------------------------------------------------

    validate_datasets(
        X_train=X_train,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation
    )

    # --------------------------------------------------------
    # Load label encoder
    # --------------------------------------------------------

    label_encoder_path = (
        MODEL_DIR / "label_encoder.joblib"
    )

    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found:\n"
            f"{label_encoder_path}\n"
            "Please run 01_prepare_ml_data.py first."
        )

    label_encoder = joblib.load(
        label_encoder_path
    )

    print("\nClass mapping:")

    for encoded_label, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"  {encoded_label} -> {class_name}"
        )

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    all_metrics = []

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Training model: {model_name}")
        print("-" * 70)

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter() - start_time
        )

        print(
            f"Training completed in "
            f"{training_time:.4f} seconds."
        )

        model_path = (
            MODEL_DIR
            / f"{model_name}_baseline.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Model saved to:\n{model_path}"
        )

        metrics = evaluate_model(
            model=model,
            model_name=model_name,
            X_validation=X_validation,
            y_validation=y_validation,
            label_encoder=label_encoder
        )

        metrics["training_time_seconds"] = (
            training_time
        )

        all_metrics.append(metrics)

    # --------------------------------------------------------
    # Compare baseline models
    # --------------------------------------------------------

    comparison_dataframe = pd.DataFrame(
        all_metrics
    )

    comparison_dataframe = (
        comparison_dataframe
        .sort_values(
            by="macro_f1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    comparison_path = (
        RESULT_DIR
        / "baseline_model_comparison.csv"
    )

    comparison_dataframe.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_time_seconds"
    ]

    print(
        comparison_dataframe[display_columns]
        .round(4)
        .to_string(index=False)
    )

    best_model_name = (
        comparison_dataframe.iloc[0]["model"]
    )

    best_macro_f1 = (
        comparison_dataframe.iloc[0]["macro_f1"]
    )

    print("\nBest baseline model:")

    print(
        f"  Model:    {best_model_name}"
    )

    print(
        f"  Macro-F1: {best_macro_f1:.4f}"
    )

    print(
        f"\nComparison results saved to:\n"
        f"{comparison_path}"
    )

    print("\nPhase 3.2 completed successfully.")


if __name__ == "__main__":
    main()