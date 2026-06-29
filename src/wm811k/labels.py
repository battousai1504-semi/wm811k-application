from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_COLUMN = "failureType_str"
UNLABELED = "unlabeled"


def extract_label(value: object) -> str:
    """Convert the nested WM-811K failureType value into one label string."""
    arr = np.asarray(value, dtype=object)

    if arr.size == 0:
        return UNLABELED

    label = arr.ravel()[0]

    if label is None:
        return UNLABELED

    return str(label)


def add_label_column(
    dataframe: pd.DataFrame,
    source_column: str = "failureType",
    target_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    result = dataframe.copy()
    result[target_column] = result[source_column].apply(extract_label)
    return result


def filter_labeled(
    dataframe: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    return dataframe[dataframe[label_column] != UNLABELED].copy()


def stratified_head(
    dataframe: pd.DataFrame,
    label_column: str,
    max_per_class: int | None,
) -> pd.DataFrame:
    if max_per_class is None:
        return dataframe.copy()

    return (
        dataframe.groupby(label_column, group_keys=False, sort=False)
        .head(max_per_class)
        .copy()
    )

