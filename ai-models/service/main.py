import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator

app = FastAPI(title='AI Service - Diabetes Prediction', version='1.0')

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_BUNDLE_PATH = MODEL_DIR / 'model.joblib'
DECISION_CONFIG_PATH = MODEL_DIR / 'decision_config.json'
MODEL_KEYS = ['logistic', 'svm', 'naive_bayes', 'random_forest']
logger = logging.getLogger(__name__)


def load_decision_config():
    try:
        with DECISION_CONFIG_PATH.open(encoding='utf-8') as config_file:
            return json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'Could not load recommendation config: {exc}')
        return {}


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
    decision_config = load_decision_config()
    if models:
        print(f'Đã load {len(models)} model(s) từ {MODEL_DIR}')
    else:
        print('Không tìm thấy model nào. Kiểm tra ai-models/models/.')
except Exception as e:
    print(f'Lỗi load model: {e}')
    scaler = None
    models = {}


class PatientData(BaseModel):
    Pregnancies: float = Field(default=0.0, validation_alias=AliasChoices('pregnancies', 'Pregnancies'))
    Glucose: float = Field(default=0.0, validation_alias=AliasChoices('glucose', 'Glucose'))
    BloodPressure: float = Field(default=0.0, validation_alias=AliasChoices('bloodPressure', 'BloodPressure'))
    SkinThickness: float = Field(default=0.0, validation_alias=AliasChoices('skinThickness', 'SkinThickness'))
    Insulin: float = Field(default=0.0, validation_alias=AliasChoices('insulin', 'Insulin'))
    BMI: float = Field(default=0.0, validation_alias=AliasChoices('bmi', 'BMI'))
    DiabetesPedigreeFunction: float = Field(default=0.0, validation_alias=AliasChoices('diabetesPedigreeFunction', 'DiabetesPedigreeFunction'))
    Age: float = Field(default=0.0, validation_alias=AliasChoices('age', 'Age'))

    @field_validator('*', mode='before')
    @classmethod
    def coerce_float(cls, value):
        if value is None:
            return 0.0
        return float(value)


@app.get('/health')
def health_check():
    return {'status': 'healthy'}


@app.get('/model-config')
def get_model_config():
    config = decision_config
    if not config or 'model_key' not in config:
        raise HTTPException(status_code=503, detail='Model config is missing. Retrain the models first.')
    if config.get('model_key') not in models:
        raise HTTPException(status_code=503, detail='Configured model is not loaded.')
    if any(key not in models for key in config.get('model_keys', [])):
        raise HTTPException(status_code=503, detail='One or more configured comparison models are not loaded.')
    if any(key not in config.get('models', {}) or 'threshold' not in config['models'][key] for key in config.get('model_keys', [])):
        raise HTTPException(status_code=503, detail='Per-model screening thresholds are missing. Retrain the models.')
    return config


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
    try:
        if hasattr(model, 'named_steps'):
            probability = float(model.predict_proba(input_df)[0][1]) if hasattr(model, 'predict_proba') else 0.0
        else:
            if model_key in ['logistic', 'svm']:
                input_processed = scaler.transform(input_df)
            else:
                input_processed = input_df
            probability = float(model.predict_proba(input_processed)[0][1]) if hasattr(model, 'predict_proba') else 0.0
    except Exception as exc:
        logger.exception('Prediction failed for model %s', model_key)
        raise HTTPException(status_code=500, detail=f'Prediction failed for model {model_key}') from exc

    try:
        threshold = float(decision_config['models'][model_key]['threshold'])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f'Screening threshold is missing for model {model_key}') from exc
    prediction = int(probability >= threshold)

    return {
        'model_used': model_key,
        'prediction': prediction,
        'probability': probability,
        'threshold': threshold,
    }
