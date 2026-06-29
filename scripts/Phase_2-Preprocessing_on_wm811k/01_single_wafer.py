import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.components import draw_components, get_connected_component_features
from wm811k.labels import LABEL_COLUMN, add_label_column
from wm811k.masks import create_defect_mask
from wm811k.paths import PHASE2_OUTPUT_DIR, RAW_DATA_PATH, ensure_dir
from wm811k.plots import save_single_wafer_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-position", type=int, default=1000)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PHASE2_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = add_label_column(pd.read_pickle(RAW_DATA_PATH))
    defect_dataframe = dataframe[dataframe["waferMap"].apply(lambda wafer: np.any(wafer == 2))]

    print("Wafers with defective dies:", len(defect_dataframe))

    sample = defect_dataframe.iloc[args.sample_position]
    wafer = sample["waferMap"]
    label = sample[LABEL_COLUMN]
    mask = create_defect_mask(wafer)
    components = get_connected_component_features(mask, min_area=args.min_area)
    component_image = draw_components(mask, components)

    print("Label:", label)
    print("Wafer shape:", wafer.shape)
    print("Wafer values:", np.unique(wafer))
    print("Components found:", len(components))
    for component in components:
        print(component)

    mask_dir = ensure_dir(args.output / "masks")
    component_dir = ensure_dir(args.output / "components")
    visualization_dir = ensure_dir(args.output / "visualizations")

    cv2.imwrite(str(mask_dir / "defect_mask.png"), mask)
    cv2.imwrite(str(component_dir / "components.png"), component_image)
    save_single_wafer_result(
        wafer=wafer,
        label=label,
        components=components,
        save_path=visualization_dir / "single_wafer_result.png",
    )

    print(f"Saved single-wafer outputs to: {args.output}")


if __name__ == "__main__":
    main()

