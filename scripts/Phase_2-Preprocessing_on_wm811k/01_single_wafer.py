import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/phase2"

os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/components", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)


def extract_label(x):
    """
    Chuyển failureType từ dạng array/list về string.
    Ví dụ:
    array([['Center']], dtype=object) -> 'Center'
    array([['none']], dtype=object)   -> 'none'
    []                               -> 'unlabeled'
    """
    arr = np.asarray(x, dtype=object)

    if arr.size == 0:
        return "unlabeled"

    value = arr.ravel()[0]

    if value is None:
        return "unlabeled"

    return str(value)


def create_defect_mask(wafer):
    """
    Tạo ảnh nhị phân defect mask.

    wafer == 2 là die lỗi.
    Ta chuyển thành:
    255 = defect
    0   = background / normal die / outside wafer
    """
    defect_mask = (wafer == 2).astype(np.uint8) * 255
    return defect_mask


def clean_mask_with_morphology(mask):
    """
    Làm sạch mask bằng morphology.

    Opening: loại bỏ defect pixel nhỏ, rời rạc.
    Closing: nối các vùng defect gần nhau.

    Với WM-811K, mỗi pixel tương ứng một die, nên kernel 3x3 là đủ.
    """
    kernel = np.ones((3, 3), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def find_connected_components(mask, min_area=2):
    """
    Tìm các vùng defect riêng biệt bằng connected components.

    min_area dùng để bỏ qua những vùng quá nhỏ.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    components = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        component = {
            "component_id": i,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy
        }

        components.append(component)

    return components


def draw_components(mask, components):
    """
    Vẽ bounding box và centroid lên ảnh mask.
    """
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for comp in components:
        x = comp["x"]
        y = comp["y"]
        w = comp["w"]
        h = comp["h"]
        cx = int(comp["cx"])
        cy = int(comp["cy"])

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.circle(
            result,
            (cx, cy),
            2,
            (255, 0, 0),
            -1
        )

    return result


def visualize_results(wafer, raw_mask, cleaned_mask, component_img, label, save_path):
    """
    Hiển thị 4 ảnh:
    1. Wafer map gốc
    2. Defect mask thô
    3. Mask sau morphology
    4. Connected components
    """
    cmap = ListedColormap(["black", "lightgray", "red"])

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(wafer, cmap=cmap, vmin=0, vmax=2)
    plt.title(f"Original\nLabel: {label}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(raw_mask, cmap="gray")
    plt.title("Raw defect mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cleaned_mask, cmap="gray")
    plt.title("After morphology")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB))
    plt.title("Components")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    df = pd.read_pickle(DATA_PATH)

    df["failureType_str"] = df["failureType"].apply(extract_label)

    print("Số lượng từng label:")
    print(df["failureType_str"].value_counts())

    # Chọn wafer có defect thật sự
    defect_df = df[df["waferMap"].apply(lambda x: np.any(x == 2))].copy()

    print("Số wafer có die lỗi:", len(defect_df))

    # Chọn thử một wafer
    sample = defect_df.iloc[1000]

    wafer = sample["waferMap"]
    label = sample["failureType_str"]

    print("Label:", label)
    print("Wafer shape:", wafer.shape)
    print("Giá trị có trong wafer:", np.unique(wafer))

    raw_mask = create_defect_mask(wafer)

    cleaned_mask = raw_mask

    components = find_connected_components(
        cleaned_mask,
        min_area=2
    )

    print("Số component tìm được:", len(components))

    for comp in components:
        print(comp)

    component_img = draw_components(cleaned_mask, components)

    cv2.imwrite(f"{OUTPUT_DIR}/masks/raw_mask.png", raw_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/masks/cleaned_mask.png", cleaned_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/components/components.png", component_img)

    visualize_results(
        wafer=wafer,
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        component_img=component_img,
        label=label,
        save_path=f"{OUTPUT_DIR}/visualizations/single_wafer_result.png"
    )


if __name__ == "__main__":
    main()