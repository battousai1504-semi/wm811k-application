import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.labels import LABEL_COLUMN, add_label_column, filter_labeled, stratified_head
from wm811k.masks import wafer_to_grayscale
from wm811k.paths import OUTPUT_DIR, RAW_DATA_PATH, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "wm811k_samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_pickle(RAW_DATA_PATH)
    dataframe = filter_labeled(add_label_column(dataframe))
    sample_dataframe = stratified_head(dataframe, LABEL_COLUMN, args.max_per_class)

    for wafer_id, row in sample_dataframe.iterrows():
        label = row[LABEL_COLUMN]
        save_dir = ensure_dir(args.output / label)
        image = wafer_to_grayscale(row["waferMap"])
        cv2.imwrite(str(save_dir / f"{label}_{wafer_id}.png"), image)

    print(f"Exported sample images to: {args.output}")


if __name__ == "__main__":
    main()

