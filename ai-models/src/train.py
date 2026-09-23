import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from preprocess import clean_and_preprocess

def main():
    # Đọc dataset (giả định file CSV nằm trong thư mục data)
    data_path = "../data/diabetes.csv" # Hoặc đường dẫn tới file giải nén của bạn
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại {data_path}. Vui lòng kiểm tra lại!")
        return

    df = pd.read_csv(data_path)
    df = clean_and_preprocess(df)

    # Chia Features và Target
    X = df.drop(columns=['Outcome', 'index'] if 'index' in df.columns else ['Outcome'])
    y = df['Outcome']

    # Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Chuẩn hóa dữ liệu cho Logistic Regression & SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Đảm bảo thư mục models tồn tại
    os.makedirs("../models", exist_ok=True)

    print("Đang huấn luyện các mô hình...")
    
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, '../models/logistic_regression_model.joblib')

    # 2. SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    joblib.dump(svm, '../models/svm_model.joblib')

    # 3. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, '../models/naive_bayes_model.joblib')

    # 4. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, '../models/random_forest_model.joblib')

    # Lưu Scaler
    joblib.dump(scaler, '../models/scaler.joblib')

    print("Huấn luyện thành công! Đã lưu toàn bộ file vào thư mục ai-models/models/")

if __name__ == "__main__":
    main()