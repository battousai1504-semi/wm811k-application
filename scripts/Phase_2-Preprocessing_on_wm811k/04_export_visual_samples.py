import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.labels import LABEL_COLUMN, add_label_column, filter_labeled, stratified_head
from wm811k.paths import PHASE2_OUTPUT_DIR, RAW_DATA_PATH
from wm811k.plots import save_class_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PHASE2_OUTPUT_DIR / "visualizations" / "class_samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = filter_labeled(add_label_column(pd.read_pickle(RAW_DATA_PATH)))
    sample_dataframe = stratified_head(dataframe, LABEL_COLUMN, args.max_per_class)

    for wafer_id, row in sample_dataframe.iterrows():
        save_class_sample(
            wafer=row["waferMap"],
            label=row[LABEL_COLUMN],
            wafer_id=wafer_id,
            output_dir=args.output,
        )

    print(f"Exported visual samples to: {args.output}")


if __name__ == "__main__":
    main()

