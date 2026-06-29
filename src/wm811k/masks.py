from __future__ import annotations

import numpy as np


def create_defect_mask(wafer: np.ndarray) -> np.ndarray:
    """Return a binary mask where defective dies are 255 and everything else is 0."""
    return (wafer == 2).astype(np.uint8) * 255


def wafer_to_grayscale(wafer: np.ndarray) -> np.ndarray:
    """Map WM-811K values 0, 1, 2 to 0, 127, 254 for image export."""
    return (wafer.astype(np.uint8) * 127)

