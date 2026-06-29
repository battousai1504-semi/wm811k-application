from __future__ import annotations

import numpy as np

from wm811k.components import get_connected_component_features, get_contour_features
from wm811k.masks import create_defect_mask


DEFAULT_FEATURE_COLUMNS = [
    "total_die",
    "defect_die",
    "defect_ratio",
    "num_components",
    "largest_component_area",
    "mean_component_area",
    "normalized_distance_to_center",
    "spread_x",
    "spread_y",
    "total_contour_area",
    "total_perimeter",
    "mean_circularity",
]


def extract_wafer_features(
    wafer: np.ndarray,
    label: str,
    wafer_id: int,
    min_component_area: int = 2,
) -> dict[str, float | int | str]:
    height, width = wafer.shape

    wafer_region = wafer > 0
    defect_region = wafer == 2

    total_die = int(np.sum(wafer_region))
    defect_die = int(np.sum(defect_region))
    defect_ratio = defect_die / total_die if total_die else 0.0

    mask = create_defect_mask(wafer)
    components = get_connected_component_features(mask, min_area=min_component_area)
    contour_features = get_contour_features(mask)

    if components:
        component_areas = [component["area"] for component in components]
        component_cx = [component["cx"] for component in components]
        component_cy = [component["cy"] for component in components]

        largest_component_area = float(np.max(component_areas))
        mean_component_area = float(np.mean(component_areas))
        mean_cx = float(np.mean(component_cx))
        mean_cy = float(np.mean(component_cy))

        center_x = width / 2
        center_y = height / 2
        distance_to_center = np.sqrt((mean_cx - center_x) ** 2 + (mean_cy - center_y) ** 2)
        diagonal = np.sqrt(width**2 + height**2)
        normalized_distance_to_center = float(distance_to_center / diagonal) if diagonal else 0.0

        spread_x = float(np.std(component_cx))
        spread_y = float(np.std(component_cy))
    else:
        largest_component_area = 0.0
        mean_component_area = 0.0
        mean_cx = 0.0
        mean_cy = 0.0
        normalized_distance_to_center = 0.0
        spread_x = 0.0
        spread_y = 0.0

    return {
        "wafer_id": wafer_id,
        "label": label,
        "height": int(height),
        "width": int(width),
        "total_die": total_die,
        "defect_die": defect_die,
        "defect_ratio": float(defect_ratio),
        "num_components": int(len(components)),
        "largest_component_area": largest_component_area,
        "mean_component_area": mean_component_area,
        "mean_cx": mean_cx,
        "mean_cy": mean_cy,
        "normalized_distance_to_center": normalized_distance_to_center,
        "spread_x": spread_x,
        "spread_y": spread_y,
        **contour_features,
    }

