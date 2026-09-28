import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

app = FastAPI(title='AI Service - Diabetes Prediction', version='1.0')

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_BUNDLE_PATH = MODEL_DIR / 'model.joblib'
MODEL_KEYS = ['logistic', 'svm', 'naive_bayes', 'random_forest']


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


def load_models():
    loaded_models = {}
    scaler = None

    if not MODEL_BUNDLE_PATH.exists():
        return scaler, loaded_models

    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except Exception as exc:
        print(f'Không thể load model bundle: {exc}')
        return scaler, loaded_models

    if isinstance(bundle, dict):
        for key in MODEL_KEYS:
            if key in bundle and bundle[key] is not None:
                loaded_models[key] = bundle[key]
        if loaded_models:
            for model in loaded_models.values():
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    scaler = model.named_steps['scaler']
                    break
        return scaler, loaded_models

    if hasattr(bundle, 'named_steps'):
        scaler = bundle.named_steps.get('scaler', scaler)
        if 'random_forest' in MODEL_KEYS:
            loaded_models['random_forest'] = bundle

    return scaler, loaded_models


try:
    scaler, models = load_models()
    if models:
        print(f'Đã load {len(models)} model(s) từ {MODEL_DIR}')
    else:
        print('Không tìm thấy model nào. Kiểm tra ai-models/models/.')
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

    @field_validator('*', mode='before')
    @classmethod
    def coerce_float(cls, value):
        if value is None:
            return 0.0
        return float(value)


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

    feature_columns = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age',
    ]

    input_df = pd.DataFrame([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
    ]], columns=feature_columns, dtype=float)

    model = models[model_key]
    if hasattr(model, 'named_steps'):
        pipeline_model = model
        prediction = int(pipeline_model.predict(input_df)[0])
        probability = float(pipeline_model.predict_proba(input_df)[0][1]) if hasattr(pipeline_model, 'predict_proba') else 0.0
    else:
        if model_key in ['logistic', 'svm']:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1]) if hasattr(model, 'predict_proba') else 0.0

    return {
        'model_used': model_key,
        'prediction': prediction,
        'probability': probability,
    }