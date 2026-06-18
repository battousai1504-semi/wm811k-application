import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/phase2/features"

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
    defect_mask = (wafer == 2).astype(np.uint8) * 255
    return defect_mask


def clean_mask_with_morphology(mask):
    kernel = np.ones((3, 3), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def get_connected_component_features(mask, min_area=2):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    component_features = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        bbox_area = w * h

        if h != 0:
            aspect_ratio = w / h
        else:
            aspect_ratio = 0

        if bbox_area != 0:
            extent = area / bbox_area
        else:
            extent = 0

        component_features.append({
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,
            "extent": extent
        })

    return component_features


def get_contour_features(mask):
    """
    Tính feature dựa trên contour:
    - perimeter
    - contour_area
    - circularity

    Circularity gần 1: hình tròn
    Circularity nhỏ: hình dài, méo, scratch-like
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {
            "total_contour_area": 0,
            "total_perimeter": 0,
            "mean_circularity": 0,
            "max_circularity": 0
        }

    contour_areas = []
    perimeters = []
    circularities = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
        else:
            circularity = 0

        contour_areas.append(contour_area)
        perimeters.append(perimeter)
        circularities.append(circularity)

    return {
        "total_contour_area": float(np.sum(contour_areas)),
        "total_perimeter": float(np.sum(perimeters)),
        "mean_circularity": float(np.mean(circularities)),
        "max_circularity": float(np.max(circularities))
    }


def extract_wafer_features(wafer, label, wafer_id):
    """
    Trích xuất feature tổng quan cho một wafer.
    """
    h, w = wafer.shape

    wafer_region = wafer > 0
    defect_region = wafer == 2

    total_die = np.sum(wafer_region)
    defect_die = np.sum(defect_region)

    if total_die > 0:
        defect_ratio = defect_die / total_die
    else:
        defect_ratio = 0

    raw_mask = create_defect_mask(wafer)
    cleaned_mask = raw_mask

    components = get_connected_component_features(
        cleaned_mask,
        min_area=2
    )

    contour_features = get_contour_features(cleaned_mask)

    num_components = len(components)

    if num_components > 0:
        component_areas = [comp["area"] for comp in components]
        largest_component_area = np.max(component_areas)
        mean_component_area = np.mean(component_areas)

        all_cx = [comp["cx"] for comp in components]
        all_cy = [comp["cy"] for comp in components]

        mean_cx = np.mean(all_cx)
        mean_cy = np.mean(all_cy)

        wafer_center_x = w / 2
        wafer_center_y = h / 2

        distance_to_center = np.sqrt(
            (mean_cx - wafer_center_x) ** 2 +
            (mean_cy - wafer_center_y) ** 2
        )

        diagonal = np.sqrt(w ** 2 + h ** 2)

        if diagonal > 0:
            normalized_distance_to_center = distance_to_center / diagonal
        else:
            normalized_distance_to_center = 0

        xs = [comp["cx"] for comp in components]
        ys = [comp["cy"] for comp in components]

        spread_x = np.std(xs)
        spread_y = np.std(ys)

    else:
        largest_component_area = 0
        mean_component_area = 0
        mean_cx = 0
        mean_cy = 0
        normalized_distance_to_center = 0
        spread_x = 0
        spread_y = 0

    features = {
        "wafer_id": wafer_id,
        "label": label,
        "height": h,
        "width": w,
        "total_die": int(total_die),
        "defect_die": int(defect_die),
        "defect_ratio": float(defect_ratio),
        "num_components": int(num_components),
        "largest_component_area": float(largest_component_area),
        "mean_component_area": float(mean_component_area),
        "mean_cx": float(mean_cx),
        "mean_cy": float(mean_cy),
        "normalized_distance_to_center": float(normalized_distance_to_center),
        "spread_x": float(spread_x),
        "spread_y": float(spread_y),
        "total_contour_area": contour_features["total_contour_area"],
        "total_perimeter": contour_features["total_perimeter"],
        "mean_circularity": contour_features["mean_circularity"],
        "max_circularity": contour_features["max_circularity"]
    }

    return features


def main():
    df = pd.read_pickle(DATA_PATH)

    df["failureType_str"] = df["failureType"].apply(extract_label)

    # Chỉ lấy wafer có label rõ ràng
    df_labeled = df[df["failureType_str"] != "unlabeled"].copy()

    print("Số wafer có label:", len(df_labeled))
    print(df_labeled["failureType_str"].value_counts())

    # Để chạy thử nhanh, mỗi class lấy tối đa 500 mẫu
    # Sau này ổn rồi có thể tăng lên hoặc bỏ dòng groupby này
    MAX_PER_CLASS = 500

    df_sample = (
        df_labeled
        .groupby("failureType_str", group_keys=False)
        .head(MAX_PER_CLASS)
        .copy()
    )

    print("Số wafer dùng để extract feature:", len(df_sample))

    all_features = []

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        wafer = row["waferMap"]
        label = row["failureType_str"]

        features = extract_wafer_features(
            wafer=wafer,
            label=label,
            wafer_id=idx
        )

        all_features.append(features)

    feature_df = pd.DataFrame(all_features)

    save_path = f"{OUTPUT_DIR}/wafer_features.csv"
    feature_df.to_csv(save_path, index=False)

    print("Đã lưu feature tại:", save_path)
    print(feature_df.head())


if __name__ == "__main__":
    main()

