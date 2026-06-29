import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.components import draw_components, get_connected_component_features
from wm811k.masks import create_defect_mask
from wm811k.paths import RAW_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-position", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_pickle(RAW_DATA_PATH)
    defect_dataframe = dataframe[dataframe["waferMap"].apply(lambda wafer: np.any(wafer == 2))]

    print("Wafers with defective dies:", len(defect_dataframe))

    wafer = defect_dataframe.iloc[args.sample_position]["waferMap"]
    mask = create_defect_mask(wafer)
    components = get_connected_component_features(mask, min_area=args.min_area)
    component_image = draw_components(mask, components)

    print("Components:", len(components))
    for component in components:
        print(component)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(wafer, cmap="gray")
    axes[0].set_title("Original waferMap")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Defect mask")
    axes[2].imshow(cv2.cvtColor(component_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Connected components")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

