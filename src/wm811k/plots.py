from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay

from wm811k.components import draw_components
from wm811k.masks import create_defect_mask
from wm811k.paths import ensure_dir


def wafer_cmap() -> ListedColormap:
    return ListedColormap(["black", "lightgray", "red"])


def show_wafer(wafer: np.ndarray, title: str) -> None:
    plt.imshow(wafer, cmap=wafer_cmap(), vmin=0, vmax=2)
    plt.title(title)
    plt.colorbar()
    plt.show()


def save_single_wafer_result(
    wafer: np.ndarray,
    label: str,
    components: list[dict[str, float]],
    save_path: Path,
) -> None:
    ensure_dir(save_path.parent)
    mask = create_defect_mask(wafer)
    component_img = draw_components(mask, components)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(wafer, cmap=wafer_cmap(), vmin=0, vmax=2)
    axes[0].set_title(f"Original\nLabel: {label}")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Defect mask")

    axes[2].imshow(cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Components")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_class_sample(
    wafer: np.ndarray,
    label: str,
    wafer_id: int,
    output_dir: Path,
) -> Path:
    label_dir = ensure_dir(output_dir / label)
    save_path = label_dir / f"{label}_{wafer_id}.png"
    mask = create_defect_mask(wafer)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(wafer, cmap=wafer_cmap(), vmin=0, vmax=2)
    axes[0].set_title(f"WaferMap\n{label}")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Defect mask")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_confusion_matrix_image(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
    values_format: str,
) -> None:
    ensure_dir(save_path.parent)
    figure, axis = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format=values_format,
        colorbar=False,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

