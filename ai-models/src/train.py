import os
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import clean_and_preprocess


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'


def resolve_data_file():
    csv_path = DATA_DIR / 'diabetes.csv'
    if csv_path.exists():
        return csv_path

    archive_path = DATA_DIR / 'archive.zip'
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        csv_path = next(DATA_DIR.glob('*.csv'), None)
        if csv_path is not None:
            return csv_path

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu. Kiểm tra {DATA_DIR} và đảm bảo có file diabetes.csv hoặc archive.zip."
    )


def main():
    data_path = resolve_data_file()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = clean_and_preprocess(df)

    feature_columns = [col for col in df.columns if col != 'Outcome' and col != 'index']
    X = df[feature_columns]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print('Đang huấn luyện các mô hình...')

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, MODEL_DIR / 'logistic_regression_model.joblib')

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    joblib.dump(svm, MODEL_DIR / 'svm_model.joblib')

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, MODEL_DIR / 'naive_bayes_model.joblib')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODEL_DIR / 'random_forest_model.joblib')

    joblib.dump(scaler, MODEL_DIR / 'scaler.joblib')

    print(f'Huấn luyện thành công! Đã lưu toàn bộ file vào {MODEL_DIR}')


if __name__ == '__main__':
    main()