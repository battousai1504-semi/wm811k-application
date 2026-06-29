import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.paths import RAW_DATA_PATH
from wm811k.plots import show_wafer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1504)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_pickle(RAW_DATA_PATH)
    wafer = dataframe.loc[args.index, "waferMap"]

    print("Wafer type:", type(wafer))
    print("Wafer shape:", wafer.shape)
    print("Wafer values:", sorted(set(wafer.flatten())))

    show_wafer(wafer, title=f"Wafer index {args.index}")


if __name__ == "__main__":
    main()

