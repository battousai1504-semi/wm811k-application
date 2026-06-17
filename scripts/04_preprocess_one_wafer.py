import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/LSWMD.pkl"

df = pd.read_pickle(DATA_PATH)

# Tìm một wafer có ít nhất một die lỗi
defect_df = df[df["waferMap"].apply(lambda x: np.any(x == 2))]

print("Số wafer có die lỗi:", len(defect_df))

wafer = defect_df.iloc[0]["waferMap"]

# 1. Tạo defect mask
defect_mask = (wafer == 2).astype("uint8") * 255

# 2. Morphology opening để loại noise nhỏ
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)

# 3. Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    opened,
    connectivity=8
)

print("Số component:", num_labels)

# 4. Vẽ bounding box
result = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

MIN_AREA = 3

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area < MIN_AREA:
        continue

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

    print(f"Component {i}: area={area}, bbox=({x}, {y}, {w}, {h})")

# 5. Hiển thị
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(wafer, cmap="gray")
plt.title("Original waferMap")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(defect_mask, cmap="gray")
plt.title("Defect mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Connected components")
plt.axis("off")

plt.show()