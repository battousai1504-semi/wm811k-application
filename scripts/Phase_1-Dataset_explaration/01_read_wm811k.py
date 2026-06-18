import pandas as pd

DaTA_PATH = "data/raw/LSWMD.pkl"
df = pd.read_pickle(DaTA_PATH)

print("Số lượng wafer:", len(df))
print("Các cột trong dataset:")
print(df.columns)

print("\n5 dòng đầu:")
print(df.head())

print("\nThông tin dataframe:")
print(df.info())