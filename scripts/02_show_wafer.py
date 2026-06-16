import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA_PATH = "data/raw/LSWMD.pkl"

df = pd.read_pickle(DATA_PATH)

# Lấy wafer đầu tiên
index = 1504
wafer = df.loc[index, "waferMap"]

print("Kiểu dữ liệu wafer:", type(wafer))
print("Kích thước wafer:", wafer.shape)
print("Giá trị có trong wafer:", set(wafer.flatten()))

# 0: không có die
# 1: die bình thường
# 2: die lỗi
cmap = ListedColormap(["black", "lightgray", "red"])

plt.imshow(wafer, cmap=cmap, vmin=0, vmax=2)
plt.title(f"Wafer index {index}")
plt.colorbar()
plt.show()