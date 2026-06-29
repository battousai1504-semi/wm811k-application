from __future__ import annotations

import cv2
import numpy as np


def get_connected_component_features(
    mask: np.ndarray,
    min_area: int = 2,
) -> list[dict[str, float]]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    components: list[dict[str, float]] = []

    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_index]
        bbox_area = width * height

        components.append(
            {
                "component_id": int(label_index),
                "area": float(area),
                "x": float(x),
                "y": float(y),
                "w": float(width),
                "h": float(height),
                "cx": float(cx),
                "cy": float(cy),
                "bbox_area": float(bbox_area),
                "aspect_ratio": float(width / height) if height else 0.0,
                "extent": float(area / bbox_area) if bbox_area else 0.0,
            }
        )

    return components


def get_contour_features(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return {
            "total_contour_area": 0.0,
            "total_perimeter": 0.0,
            "mean_circularity": 0.0,
            "max_circularity": 0.0,
        }

    contour_areas = []
    perimeters = []
    circularities = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * contour_area) / (perimeter**2) if perimeter else 0.0

        contour_areas.append(contour_area)
        perimeters.append(perimeter)
        circularities.append(circularity)

    return {
        "total_contour_area": float(np.sum(contour_areas)),
        "total_perimeter": float(np.sum(perimeters)),
        "mean_circularity": float(np.mean(circularities)),
        "max_circularity": float(np.max(circularities)),
    }


def draw_components(mask: np.ndarray, components: list[dict[str, float]]) -> np.ndarray:
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        width = int(component["w"])
        height = int(component["h"])
        cx = int(component["cx"])
        cy = int(component["cy"])

        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.circle(result, (cx, cy), 2, (255, 0, 0), -1)

    return result

