from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


# ============================================================
# 2. SUPPORT FUNCTIONS
# ============================================================

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("PHASE 3.1 — MACHINE LEARNING DATA PREPARATION")
    print("=" * 60)

    print(f"\nDataset path: {DATA_PATH}")
    print(f"Dataset shape: {dataframe.shape}")

    print("\nDataset columns:")
    for column in dataframe.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Identify label and feature columns
    # --------------------------------------------------------

    label_column = find_label_column(dataframe)

    non_feature_columns = {
        label_column,
        "wafer_id",
        "waferId",
        "image_name",
        "filename",
        "file_path",
        "path"
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in non_feature_columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if len(feature_columns) == 0:
        raise ValueError("No numerical feature columns were found.")

    print(f"\nLabel column: {label_column}")
    print(f"Number of features: {len(feature_columns)}")

    print("\nSelected feature columns:")
    for feature in feature_columns:
        print(f"  - {feature}")

    if len(feature_columns) != 12:
        print(
            "\nWarning: The program did not detect exactly "
            f"12 features. Detected: {len(feature_columns)}"
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
        OUTPUT_DIR / "X_train.csv",
        index=False
    )

    X_validation_scaled.to_csv(
        OUTPUT_DIR / "X_validation.csv",
        index=False
    )

    X_test_scaled.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False
    )

    pd.DataFrame({"label": y_train}).to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False
    )

    pd.DataFrame({"label": y_validation}).to_csv(
        OUTPUT_DIR / "y_validation.csv",
        index=False
    )

    pd.DataFrame({"label": y_test}).to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False
    )

    # Save preprocessing objects
    joblib.dump(
        scaler,
        MODEL_DIR / "standard_scaler.joblib"
    )

    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.joblib"
    )

    joblib.dump(
        feature_columns,
        MODEL_DIR / "feature_columns.joblib"
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

    print(f"\nSaved dataset splits to:\n{OUTPUT_DIR}")
    print(f"\nSaved preprocessing objects to:\n{MODEL_DIR}")

    print("\nPhase 3.1 completed successfully.")


if __name__ == "__main__":
    main()