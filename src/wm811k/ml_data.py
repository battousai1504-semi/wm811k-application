from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def find_label_column(dataframe: pd.DataFrame) -> str:
    for column in (
        "label",
        "class",
        "faultType",
        "failureType",
        "defect_type",
        "defect_class",
        "target",
    ):
        if column in dataframe.columns:
            return column

    raise ValueError(f"Cannot find a label column. Available columns: {list(dataframe.columns)}")


def infer_numeric_feature_columns(
    dataframe: pd.DataFrame,
    label_column: str,
) -> list[str]:
    non_feature_columns = {
        label_column,
        "wafer_id",
        "waferId",
        "image_name",
        "filename",
        "file_path",
        "path",
    }

    return [
        column
        for column in dataframe.columns
        if column not in non_feature_columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def clean_invalid_rows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, int]:
    cleaned = dataframe.copy()
    cleaned[feature_columns] = cleaned[feature_columns].replace([np.inf, -np.inf], np.nan)
    rows_before = len(cleaned)
    cleaned = cleaned.dropna(subset=feature_columns + [label_column])
    return cleaned, rows_before - len(cleaned)


def display_class_distribution(labels: pd.Series, title: str) -> None:
    counts = labels.value_counts()
    percentages = labels.value_counts(normalize=True) * 100
    distribution = pd.DataFrame(
        {
            "sample_count": counts,
            "percentage": percentages.round(2),
        }
    )
    print(f"\n{title}")
    print(distribution.sort_index())


def split_and_scale(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
):
    X = dataframe[feature_columns].copy()
    y_text = dataframe[label_column].astype(str).copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=y,
    )

    relative_test_size = test_size / (test_size + validation_size)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )
    X_validation_scaled = pd.DataFrame(
        scaler.transform(X_validation),
        columns=feature_columns,
        index=X_validation.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
        index=X_test.index,
    )

    return {
        "X_train": X_train_scaled,
        "X_validation": X_validation_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }


def save_phase3_outputs(
    split_data: dict,
    feature_columns: list[str],
    split_dir,
    model_dir,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    split_data["X_train"].to_csv(split_dir / "X_train.csv", index=False)
    split_data["X_validation"].to_csv(split_dir / "X_validation.csv", index=False)
    split_data["X_test"].to_csv(split_dir / "X_test.csv", index=False)
    pd.DataFrame({"label": split_data["y_train"]}).to_csv(split_dir / "y_train.csv", index=False)
    pd.DataFrame({"label": split_data["y_validation"]}).to_csv(split_dir / "y_validation.csv", index=False)
    pd.DataFrame({"label": split_data["y_test"]}).to_csv(split_dir / "y_test.csv", index=False)

    joblib.dump(split_data["scaler"], model_dir / "standard_scaler.joblib")
    joblib.dump(split_data["label_encoder"], model_dir / "label_encoder.joblib")
    joblib.dump(feature_columns, model_dir / "feature_columns.joblib")

