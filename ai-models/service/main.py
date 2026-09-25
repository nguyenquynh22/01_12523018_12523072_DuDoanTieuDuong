import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator

app = FastAPI(title='AI Service - Diabetes Prediction', version='1.0')

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace(' ', '_').replace('-', '_')
    aliases = {
        'logistic': 'logistic',
        'logistic_regression': 'logistic',
        'svm': 'svm',
        'svc': 'svm',
        'naive_bayes': 'naive_bayes',
        'naivebayes': 'naive_bayes',
        'random_forest': 'random_forest',
        'randomforest': 'random_forest',
    }
    return aliases.get(normalized, normalized)


try:
    scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
    models = {
        'logistic': joblib.load(MODEL_DIR / 'logistic_regression_model.joblib'),
        'svm': joblib.load(MODEL_DIR / 'svm_model.joblib'),
        'naive_bayes': joblib.load(MODEL_DIR / 'naive_bayes_model.joblib'),
        'random_forest': joblib.load(MODEL_DIR / 'random_forest_model.joblib'),
    }
except Exception as e:
    print(f'Lỗi load model: {e}')
    scaler = None
    models = {}


class PatientData(BaseModel):
    Pregnancies: float = 0.0
    Glucose: float = 0.0
    BloodPressure: float = 0.0
    SkinThickness: float = 0.0
    Insulin: float = 0.0
    BMI: float = 0.0
    DiabetesPedigreeFunction: float = 0.0
    Age: float = 0.0

    @root_validator(pre=True)
    def normalize_input_keys(cls, values):
        if not isinstance(values, dict):
            return values

        normalized = {}
        field_map = {
            'pregnancies': 'Pregnancies',
            'glucose': 'Glucose',
            'bloodpressure': 'BloodPressure',
            'skinthickness': 'SkinThickness',
            'insulin': 'Insulin',
            'bmi': 'BMI',
            'diabetespedigreefunction': 'DiabetesPedigreeFunction',
            'age': 'Age',
        }

        for key, value in values.items():
            mapped_key = field_map.get(str(key).lower(), key)
            normalized[mapped_key] = value

        return normalized


@app.get('/health')
def health_check():
    return {'status': 'healthy'}


@app.post('/predict/{model_name}')
def predict(model_name: str, data: PatientData):
    if not models or scaler is None:
        raise HTTPException(status_code=503, detail='Model chưa được huấn luyện hoặc file model bị thiếu.')

    model_key = normalize_model_name(model_name)
    if model_key not in models:
        raise HTTPException(
            status_code=400,
            detail='Model không tồn tại. Chọn: logistic, svm, naive_bayes, random_forest',
        )

    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
    ]], dtype=float)

    model = models[model_key]
    if model_key in ['logistic', 'svm']:
        input_processed = scaler.transform(input_data)
    else:
        input_processed = input_data

    prediction = int(model.predict(input_processed)[0])
    probability = float(model.predict_proba(input_processed)[0][1]) if hasattr(model, 'predict_proba') else 0.0

    return {
        'model_used': model_key,
        'prediction': prediction,
        'probability': probability,
    }