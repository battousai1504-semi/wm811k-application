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

LBP_FEATURE_COLUMNS = [f"lbp_{index:03d}" for index in range(256)]

GLCM_STATISTICS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "asm",
    "energy",
    "correlation",
]
GLCM_FEATURE_COLUMNS = [
    f"glcm_{statistic}_{summary}"
    for statistic in GLCM_STATISTICS
    for summary in ("mean", "std")
]

TEXTURE_FEATURE_COLUMNS = LBP_FEATURE_COLUMNS + GLCM_FEATURE_COLUMNS
CLASSICAL_CV_FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS + TEXTURE_FEATURE_COLUMNS


def calculate_lbp_histogram(wafer: np.ndarray) -> dict[str, float]:
    image = np.asarray(wafer, dtype=np.uint8)
    if image.shape[0] < 3 or image.shape[1] < 3:
        return {column: 0.0 for column in LBP_FEATURE_COLUMNS}

    center = image[1:-1, 1:-1]
    neighbors = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2],
    ]

    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)

    histogram = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total:
        histogram /= total

    return {
        column: float(histogram[index])
        for index, column in enumerate(LBP_FEATURE_COLUMNS)
    }


def _offset_views(
    image: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape

    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))

    return image[source_y, source_x], image[target_y, target_x]


def _normalized_glcm(
    image: np.ndarray,
    dy: int,
    dx: int,
    levels: int = 3,
) -> np.ndarray:
    source, target = _offset_views(image, dy, dx)
    matrix = np.zeros((levels, levels), dtype=np.float64)

    if source.size == 0 or target.size == 0:
        return matrix

    np.add.at(matrix, (source.ravel(), target.ravel()), 1)
    np.add.at(matrix, (target.ravel(), source.ravel()), 1)

    total = matrix.sum()
    if total:
        matrix /= total

    return matrix


def _glcm_statistics(matrix: np.ndarray) -> dict[str, float]:
    levels = matrix.shape[0]
    rows, columns = np.indices((levels, levels))
    difference = rows - columns

    contrast = np.sum(matrix * difference**2)
    dissimilarity = np.sum(matrix * np.abs(difference))
    homogeneity = np.sum(matrix / (1.0 + difference**2))
    asm = np.sum(matrix**2)
    energy = np.sqrt(asm)

    row_mean = np.sum(rows * matrix)
    column_mean = np.sum(columns * matrix)
    row_var = np.sum(((rows - row_mean) ** 2) * matrix)
    column_var = np.sum(((columns - column_mean) ** 2) * matrix)
    denominator = np.sqrt(row_var * column_var)
    correlation = (
        np.sum((rows - row_mean) * (columns - column_mean) * matrix) / denominator
        if denominator > 0
        else 0.0
    )

    return {
        "contrast": float(contrast),
        "dissimilarity": float(dissimilarity),
        "homogeneity": float(homogeneity),
        "asm": float(asm),
        "energy": float(energy),
        "correlation": float(correlation),
    }


def calculate_glcm_features(wafer: np.ndarray) -> dict[str, float]:
    image = np.asarray(wafer, dtype=np.int16)
    image = np.clip(image, 0, 2).astype(np.uint8)
    offsets = [
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, 2),
        (-2, 2),
        (-2, 0),
        (-2, -2),
    ]

    values_by_statistic = {statistic: [] for statistic in GLCM_STATISTICS}
    for dy, dx in offsets:
        matrix = _normalized_glcm(image, dy=dy, dx=dx, levels=3)
        statistics = _glcm_statistics(matrix)
        for statistic, value in statistics.items():
            values_by_statistic[statistic].append(value)

    features = {}
    for statistic, values in values_by_statistic.items():
        array = np.asarray(values, dtype=np.float64)
        features[f"glcm_{statistic}_mean"] = float(array.mean())
        features[f"glcm_{statistic}_std"] = float(array.std(ddof=0))

    return features


def extract_texture_features(wafer: np.ndarray) -> dict[str, float]:
    return {
        **calculate_lbp_histogram(wafer),
        **calculate_glcm_features(wafer),
    }


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
        **extract_texture_features(wafer),
    }

