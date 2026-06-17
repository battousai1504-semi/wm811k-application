import os
import pandas as pd
import numpy as np
import cv2

DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/wm811k_samples"

df = pd.read_pickle(DATA_PATH)

def extract_label(x):
    arr = np.asarray(x)

    if arr.size == 0:
        return "unlabeled"

    arr = arr.ravel()

    if len(arr) == 0:
        return "unlabeled"

    return str(arr[0])

df["failureType_str"] = df["failureType"].apply(extract_label)

# Bỏ mẫu chưa có nhãn
df_labeled = df[df["failureType_str"] != "unlabeled"].copy()

# Mỗi class lấy tối đa 100 ảnh để test trước
MAX_PER_CLASS = 100

for label, group in df_labeled.groupby("failureType_str"):
    save_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    sample_group = group.head(MAX_PER_CLASS)

    for idx, row in sample_group.iterrows():
        wafer = row["waferMap"]

        # Chuyển 0,1,2 thành 0,127,255 để lưu ảnh grayscale
        img = (wafer.astype("uint8") * 127)

        filename = f"{label}_{idx}.png"
        path = os.path.join(save_dir, filename)

        cv2.imwrite(path, img)

print("Đã export xong ảnh mẫu.")