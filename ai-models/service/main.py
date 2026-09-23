from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="AI Service - Diabetes Prediction", version="1.0")

# Đường dẫn tới thư mục models (nằm ngoài service một cấp)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Load sẵn các model và scaler khi container khởi động
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    models = {
        "logistic": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.joblib")),
        "svm": joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib")),
        "naive_bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes_model.joblib")),
        "random_forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib")),
    }
except Exception as e:
    print(f"Lỗi load model: {e}")

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/{model_name}")
def predict(model_name: str, data: PatientData):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model không tồn tại. Chọn: logistic, svm, naive_bayes, random_forest")
    
    # Chuyển dữ liệu đầu vào thành mảng numpy
    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure, 
        data.SkinThickness, data.Insulin, data.BMI, 
        data.DiabetesPedigreeFunction, data.Age
    ]])
    
    model = models[model_name]
    
    # Logistic và SVM cần scale dữ liệu, Naive Bayes và Random Forest thì không
    if model_name in ["logistic", "svm"]:
        input_processed = scaler.transform(input_data)
    else:
        input_processed = input_data
        
    prediction = int(model.predict(input_processed)[0])
    
    # Lấy xác suất nếu model hỗ trợ
    probability = float(model.predict_proba(input_processed)[0][1]) if hasattr(model, "predict_proba") else 0.0

    return {
        "model_used": model_name,
        "prediction": prediction,
        "probability": probability
    }