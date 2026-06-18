import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/phase2/visualizations/class_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_label(x):
    arr = np.asarray(x, dtype=object)

    if arr.size == 0:
        return "unlabeled"

    value = arr.ravel()[0]

    if value is None:
        return "unlabeled"

    return str(value)


def create_defect_mask(wafer):
    return (wafer == 2).astype(np.uint8) * 255


def save_visual_sample(wafer, label, wafer_id):
    cmap = ListedColormap(["black", "lightgray", "red"])
    mask = create_defect_mask(wafer)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(wafer, cmap=cmap, vmin=0, vmax=2)
    plt.title(f"WaferMap\n{label}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Defect Mask")
    plt.axis("off")

    plt.tight_layout()

    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    save_path = os.path.join(label_dir, f"{label}_{wafer_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    df = pd.read_pickle(DATA_PATH)

    df["failureType_str"] = df["failureType"].apply(extract_label)

    df_labeled = df[df["failureType_str"] != "unlabeled"].copy()

    MAX_PER_CLASS = 5

    df_sample = (
        df_labeled
        .groupby("failureType_str", group_keys=False)
        .head(MAX_PER_CLASS)
        .copy()
    )

    for idx, row in df_sample.iterrows():
        wafer = row["waferMap"]
        label = row["failureType_str"]

        save_visual_sample(
            wafer=wafer,
            label=label,
            wafer_id=idx
        )

    print("Đã export ảnh sample cho từng class.")


if __name__ == "__main__":
    main()