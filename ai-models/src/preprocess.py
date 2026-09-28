import numpy as np
import pandas as pd


def clean_and_preprocess(df):
    """
    Làm sạch dữ liệu: thay các giá trị 0 không hợp lệ bằng NaN ở các cột y tế
    và điền bằng median để giảm ảnh hưởng outlier.
    """
    df = df.copy()
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in cols_to_fix:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
        median_value = df[col].median()
        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

    return df