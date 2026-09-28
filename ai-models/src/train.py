import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
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

    for archive_name in ('archive.zip',):
        archive_path = DATA_DIR / archive_name
        if archive_path.exists():
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            csv_path = next(DATA_DIR.glob('*.csv'), None)
            if csv_path is not None:
                return csv_path

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu. Kiểm tra {DATA_DIR} và đảm bảo có file diabetes.csv hoặc archive.zip."
    )


def build_pipeline(model_name: str, estimator):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', estimator),
    ])


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

    print('Đang huấn luyện các mô hình...')

    models = {
        'logistic': build_pipeline('logistic', LogisticRegression(random_state=42, max_iter=1000)),
        'svm': build_pipeline('svm', SVC(probability=True, random_state=42)),
        'naive_bayes': Pipeline([('scaler', StandardScaler()), ('classifier', GaussianNB())]),
        'random_forest': Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
    }

    for name, model in models.items():
        if name in {'logistic', 'svm'}:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

    joblib.dump(models, MODEL_DIR / 'model.joblib', compress=3)

    print(f'Huấn luyện thành công! Đã lưu bundle model.joblib vào {MODEL_DIR}')


if __name__ == '__main__':
    main()