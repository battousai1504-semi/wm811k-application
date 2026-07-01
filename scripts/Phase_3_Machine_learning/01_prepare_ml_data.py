import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.features import DEFAULT_FEATURE_COLUMNS

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wafer_features.csv"
)

OUTPUT_DIR = (
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

FEATURE_SELECTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
    / "feature_analysis"
    / "05_selected_features.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


# ============================================================
# 2. SUPPORT FUNCTIONS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--feature-set",
        choices=["auto", "selected", "default"],
        default="auto",
        help=(
            "Feature set used for ML. 'auto' uses the correlation-selected "
            "features if they exist, otherwise the default handcrafted features."
        ),
    )
    parser.add_argument(
        "--selected-features",
        type=Path,
        default=FEATURE_SELECTION_PATH,
        help="CSV produced by 05_feature_analysis.py.",
    )
    return parser.parse_args()


def find_label_column(dataframe: pd.DataFrame) -> str:
    """
    Automatically search for the class-label column.

    The function checks several common label-column names.
    """

    possible_label_columns = [
        "label",
        "class",
        "faultType",
        "failureType",
        "defect_type",
        "defect_class",
        "target"
    ]

    for column in possible_label_columns:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Cannot find the label column.\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def load_selected_feature_columns(path: Path) -> list[str]:
    selected_dataframe = pd.read_csv(path)
    if "selected_feature" not in selected_dataframe.columns:
        raise ValueError(
            "Selected-feature file must contain a 'selected_feature' column:\n"
            f"{path}"
        )

    features = (
        selected_dataframe["selected_feature"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if not features:
        raise ValueError(f"No selected features were found in:\n{path}")

    return features


def validate_feature_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    missing = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]
    non_numeric = [
        feature
        for feature in feature_columns
        if feature in dataframe.columns
        and not pd.api.types.is_numeric_dtype(dataframe[feature])
    ]

    if missing or non_numeric:
        raise ValueError(
            "Invalid feature columns.\n"
            f"Missing: {missing}\n"
            f"Non-numeric: {non_numeric}"
        )


def choose_feature_columns(
    dataframe: pd.DataFrame,
    feature_set: str,
    selected_features_path: Path,
) -> tuple[list[str], str]:
    if feature_set in {"auto", "selected"} and selected_features_path.exists():
        selected_features = load_selected_feature_columns(selected_features_path)
        validate_feature_columns(dataframe, selected_features)
        return (
            selected_features,
            f"correlation-selected features from {selected_features_path}",
        )

    if feature_set == "selected":
        raise FileNotFoundError(
            "Selected-feature file not found. Run 05_feature_analysis.py first:\n"
            f"{selected_features_path}"
        )

    validate_feature_columns(dataframe, DEFAULT_FEATURE_COLUMNS)
    return (
        list(DEFAULT_FEATURE_COLUMNS),
        "default handcrafted feature list",
    )


def remove_invalid_rows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str
) -> pd.DataFrame:
    """
    Remove rows containing missing or infinite values.
    """

    cleaned_dataframe = dataframe.copy()

    cleaned_dataframe[feature_columns] = (
        cleaned_dataframe[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    rows_before = len(cleaned_dataframe)

    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=feature_columns + [label_column]
    )

    rows_after = len(cleaned_dataframe)

    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning:  {rows_after}")
    print(f"Removed rows:         {rows_before - rows_after}")

    return cleaned_dataframe


def display_class_distribution(
    labels: pd.Series,
    title: str
) -> None:
    """
    Print the number and percentage of samples in each class.
    """

    counts = labels.value_counts()
    percentages = labels.value_counts(normalize=True) * 100

    distribution = pd.DataFrame({
        "sample_count": counts,
        "percentage": percentages.round(2)
    })

    print(f"\n{title}")
    print(distribution.sort_index())


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def main() -> None:
    """
    Prepare training, validation, and test datasets.
    """

    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{args.data_path}"
        )

    dataframe = pd.read_csv(args.data_path)

    print("=" * 60)
    print("PHASE 3.1 - MACHINE LEARNING DATA PREPARATION")
    print("=" * 60)

    print(f"\nDataset path: {args.data_path}")
    print(f"Dataset shape: {dataframe.shape}")

    print("\nDataset columns:")
    for column in dataframe.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Identify label and feature columns
    # --------------------------------------------------------

    label_column = find_label_column(dataframe)
    feature_columns, feature_source = choose_feature_columns(
        dataframe=dataframe,
        feature_set=args.feature_set,
        selected_features_path=args.selected_features,
    )

    if len(feature_columns) == 0:
        raise ValueError("No numerical feature columns were found.")

    print(f"\nLabel column: {label_column}")
    print(f"Feature source: {feature_source}")
    print(f"Number of features: {len(feature_columns)}")

    print("\nSelected feature columns:")
    for feature in feature_columns:
        print(f"  - {feature}")

    if len(feature_columns) != len(DEFAULT_FEATURE_COLUMNS):
        print(
            "\nNote: The selected feature set contains "
            f"{len(feature_columns)} features. The default handcrafted set "
            f"contains {len(DEFAULT_FEATURE_COLUMNS)} features."
        )

    # --------------------------------------------------------
    # Clean invalid data
    # --------------------------------------------------------

    dataframe = remove_invalid_rows(
        dataframe=dataframe,
        feature_columns=feature_columns,
        label_column=label_column
    )

    display_class_distribution(
        dataframe[label_column],
        "Complete dataset class distribution"
    )

    # --------------------------------------------------------
    # Create X and y
    # --------------------------------------------------------

    X = dataframe[feature_columns].copy()
    y_text = dataframe[label_column].astype(str).copy()

    # Convert text labels into integer labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    print("\nLabel encoding:")

    for encoded_value, class_name in enumerate(
        label_encoder.classes_
    ):
        print(f"  {class_name} -> {encoded_value}")

    # --------------------------------------------------------
    # Split train and temporary datasets
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Temporary dataset contains 30% of the samples.
    # It is divided equally into validation and test sets.

    relative_test_size = (
        TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
    )

    # --------------------------------------------------------
    # Standardize numerical features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back into DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_columns,
        index=X_train.index
    )

    X_validation_scaled = pd.DataFrame(
        X_validation_scaled,
        columns=feature_columns,
        index=X_validation.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_columns,
        index=X_test.index
    )

    # --------------------------------------------------------
    # Save the datasets
    # --------------------------------------------------------

    X_train_scaled.to_csv(
        args.output_dir / "X_train.csv",
        index=False
    )

    X_validation_scaled.to_csv(
        args.output_dir / "X_validation.csv",
        index=False
    )

    X_test_scaled.to_csv(
        args.output_dir / "X_test.csv",
        index=False
    )

    pd.DataFrame({"label": y_train}).to_csv(
        args.output_dir / "y_train.csv",
        index=False
    )

    pd.DataFrame({"label": y_validation}).to_csv(
        args.output_dir / "y_validation.csv",
        index=False
    )

    pd.DataFrame({"label": y_test}).to_csv(
        args.output_dir / "y_test.csv",
        index=False
    )

    # Save preprocessing objects
    joblib.dump(
        scaler,
        args.model_dir / "standard_scaler.joblib"
    )

    joblib.dump(
        label_encoder,
        args.model_dir / "label_encoder.joblib"
    )

    joblib.dump(
        feature_columns,
        args.model_dir / "feature_columns.joblib"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)

    print(f"Training samples:   {len(X_train_scaled)}")
    print(f"Validation samples: {len(X_validation_scaled)}")
    print(f"Test samples:       {len(X_test_scaled)}")

    display_class_distribution(
        pd.Series(
            label_encoder.inverse_transform(y_train)
        ),
        "Training-set class distribution"
    )

    display_class_distribution(
        pd.Series(
            label_encoder.inverse_transform(y_validation)
        ),
        "Validation-set class distribution"
    )

    display_class_distribution(
        pd.Series(
            label_encoder.inverse_transform(y_test)
        ),
        "Test-set class distribution"
    )

    print(f"\nSaved dataset splits to:\n{args.output_dir}")
    print(f"\nSaved preprocessing objects to:\n{args.model_dir}")

    print("\nPhase 3.1 completed successfully.")


if __name__ == "__main__":
    main()
