import pandas as pd
import numpy as np

DATA_PATH = "data/raw/LSWMD.pkl"

df = pd.read_pickle(DATA_PATH)

def extract_label(x):
    """
    Chuyển label từ dạng array/list về string.
    Ví dụ:
    array(['Center']) -> 'Center'
    array(['none'])   -> 'none'
    []                -> 'unlabeled'
    """
    arr = np.asarray(x)

    if arr.size == 0:
        return "unlabeled"

    arr = arr.ravel()

    if len(arr) == 0:
        return "unlabeled"

    return str(arr[0])

df["failureType_str"] = df["failureType"].apply(extract_label)

print(df["failureType_str"].value_counts())