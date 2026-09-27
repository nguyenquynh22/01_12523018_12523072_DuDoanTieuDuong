import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pymongo import MongoClient
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Web Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def normalize_trailing_dots(request: Request, call_next):
    path = request.url.path
    if path.endswith("."):
        normalized_path = path.rstrip(".") or "/"
        return RedirectResponse(normalized_path, status_code=307)

    if path == "/favicon.ico":
        return Response(status_code=204)

    return await call_next(request)


def resolve_ai_service_url():
    url = os.getenv("AI_SERVICE_URL", "http://localhost:8001").rstrip("/")
    parsed = urlparse(url)
    host = parsed.hostname

    if host and host not in {"localhost", "127.0.0.1"}:
        try:
            socket.getaddrinfo(host, parsed.port or 80)
        except socket.gaierror:
            return url.replace(host, "localhost", 1)

    return url


AI_SERVICE_URL = resolve_ai_service_url()
MODEL_NAMES = ["logistic", "svm", "naive_bayes", "random_forest"]
MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "svm": "Support Vector Machine (SVM)",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
}

MONGODB_URI = os.getenv("MONGODB_URI")


def init_mongo_client():
    if not MONGODB_URI:
        print("MongoDB URI không được cấu hình. Lịch sử dự đoán sẽ không được lưu.")
        return None

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        print("MongoDB kết nối thành công. Lịch sử dự đoán sẽ được lưu.")
        return client
    except Exception as exc:
        print(f"MongoDB không khả dụng, bỏ qua lưu lịch sử: {exc}")
        return None


mongo_client = init_mongo_client()


@app.on_event("shutdown")
def shutdown_mongo_client():
    if mongo_client is not None:
        mongo_client.close()


class PatientInput(BaseModel):
    pregnancies: float
    glucose: float
    bloodPressure: float
    skinThickness: float
    insulin: float
    bmi: float
    diabetesPedigreeFunction: float
    age: float


@app.get("/health")
def health_check():
    return {"status": "ok", "ai_service_url": AI_SERVICE_URL}


@app.post("/api/v1/predict-disease")
def proxy_to_ai_service(data: PatientInput):
    payload = data.model_dump()
    results = []

    try:
        for model_name in MODEL_NAMES:
            response = requests.post(
                f"{AI_SERVICE_URL}/predict/{model_name}",
                json=payload,
                timeout=20,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"AI Service từ chối xử lý model {model_name}",
                )

            result = response.json()
            probability = float(result.get("probability", 0.0)) * 100
            prediction = int(result.get("prediction", 0))

            results.append({
                "modelName": MODEL_LABELS[model_name],
                "prediction": prediction,
                "probability": round(probability, 1),
                "statusText": "Nguy cơ tiểu đường" if prediction == 1 else "Bình thường",
                "description": "Mô hình AI được đánh giá theo chỉ số sức khỏe bệnh nhân.",
            })

        best_model = max(results, key=lambda item: item["probability"])
        now_utc = datetime.now(timezone.utc)
        response_payload = {
            "id": f"pred-{now_utc.strftime('%Y%m%d%H%M%S')}",
            "createdAt": now_utc.strftime("%d/%m/%Y %H:%M:%S"),
            "inputData": {
                "pregnancies": payload["pregnancies"],
                "glucose": payload["glucose"],
                "bloodPressure": payload["bloodPressure"],
                "skinThickness": payload["skinThickness"],
                "insulin": payload["insulin"],
                "bmi": payload["bmi"],
                "diabetesPedigreeFunction": payload["diabetesPedigreeFunction"],
                "age": payload["age"],
            },
            "results": results,
            "bestModel": best_model["modelName"],
        }

        if mongo_client is not None:
            database = mongo_client.get_default_database()
            prediction_history = database["prediction_history"]
            prediction_history.insert_one({
                "request_id": response_payload["id"],
                "created_at": now_utc,
                "input_data": response_payload["inputData"],
                "results": response_payload["results"],
                "best_model": response_payload["bestModel"],
            })

        return response_payload

    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối đến AI-Service tại {AI_SERVICE_URL}. Chi tiết: {str(exc)}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dự đoán: {str(exc)}")