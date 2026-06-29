from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.labels import LABEL_COLUMN, add_label_column
from wm811k.paths import RAW_DATA_PATH


def main() -> None:
    dataframe = pd.read_pickle(RAW_DATA_PATH)
    dataframe = add_label_column(dataframe)
    print(dataframe[LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    main()

