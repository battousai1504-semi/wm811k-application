from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.paths import RAW_DATA_PATH


def main() -> None:
    dataframe = pd.read_pickle(RAW_DATA_PATH)

    print("Number of wafers:", len(dataframe))
    print("\nDataset columns:")
    print(dataframe.columns)

    print("\nFirst 5 rows:")
    print(dataframe.head())

    print("\nDataFrame info:")
    dataframe.info()


if __name__ == "__main__":
    main()

