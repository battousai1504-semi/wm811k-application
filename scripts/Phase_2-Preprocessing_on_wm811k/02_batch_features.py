import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.features import extract_wafer_features
from wm811k.labels import LABEL_COLUMN, add_label_column, filter_labeled, stratified_head
from wm811k.paths import FEATURE_DATA_PATH, RAW_DATA_PATH, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help=(
            "Maximum labeled wafers to extract per class. "
            "Omit this option to use every labeled wafer in WM-811K."
        ),
    )
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--output", type=Path, default=FEATURE_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = add_label_column(pd.read_pickle(RAW_DATA_PATH))
    labeled_dataframe = filter_labeled(dataframe, label_column=LABEL_COLUMN)
    sample_dataframe = stratified_head(
        labeled_dataframe,
        label_column=LABEL_COLUMN,
        max_per_class=args.max_per_class,
    )

    print("Labeled wafers:", len(labeled_dataframe))
    print(labeled_dataframe[LABEL_COLUMN].value_counts())
    print(
        "Max wafers per class:",
        "all labeled wafers" if args.max_per_class is None else args.max_per_class,
    )
    print("Wafers used for feature extraction:", len(sample_dataframe))

    rows = []
    for wafer_id, row in tqdm(sample_dataframe.iterrows(), total=len(sample_dataframe)):
        rows.append(
            extract_wafer_features(
                wafer=row["waferMap"],
                label=row[LABEL_COLUMN],
                wafer_id=wafer_id,
                min_component_area=args.min_area,
            )
        )

    feature_dataframe = pd.DataFrame(rows)
    ensure_dir(args.output.parent)
    feature_dataframe.to_csv(args.output, index=False)

    print(f"Saved features to: {args.output}")
    print(feature_dataframe.head())


if __name__ == "__main__":
    main()

