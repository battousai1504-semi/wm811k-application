import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.paths import FEATURE_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=FEATURE_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_csv(args.csv)

    print("Feature table shape:")
    print(dataframe.shape)

    print("\nFirst 5 rows:")
    print(dataframe.head())

    print("\nBasic statistics:")
    print(dataframe.describe())

    print("\nLabel counts:")
    print(dataframe["label"].value_counts())

    print("\nMean defect ratio by label:")
    print(dataframe.groupby("label")["defect_ratio"].mean().sort_values(ascending=False))

    print("\nMean component count by label:")
    print(dataframe.groupby("label")["num_components"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()

