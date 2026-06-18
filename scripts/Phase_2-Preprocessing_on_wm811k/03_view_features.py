import pandas as pd

FEATURE_PATH = "outputs/phase2/features/wafer_features.csv"

df = pd.read_csv(FEATURE_PATH)

print("Kích thước bảng feature:")
print(df.shape)

print("\n5 dòng đầu:")
print(df.head())

print("\nThống kê cơ bản:")
print(df.describe())

print("\nSố lượng label:")
print(df["label"].value_counts())

print("\nDefect ratio trung bình theo label:")
print(df.groupby("label")["defect_ratio"].mean().sort_values(ascending=False))

print("\nSố component trung bình theo label:")
print(df.groupby("label")["num_components"].mean().sort_values(ascending=False))