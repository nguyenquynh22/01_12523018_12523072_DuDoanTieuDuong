import pandas as pd
import numpy as np

def clean_and_preprocess(df):
    """
    Hàm làm sạch dữ liệu: Thay thế giá trị 0 bằng NaN ở các cột y tế 
    và điền giá trị thiếu bằng Median để tránh ảnh hưởng outlier.
    """
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())
        
    return df