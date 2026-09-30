import os
from concurrent.futures import ThreadPoolExecutor
import logging
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

BASE_DIR = Path(__file__).resolve().parent
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(PROJECT_ROOT / ".env")
except IndexError:
    # Trường hợp chạy trong Docker hoặc cấu trúc nông hơn
    load_dotenv(BASE_DIR / ".env")

load_dotenv()

app = FastAPI(title="Web Backend API", version="1.0.0")
logger = logging.getLogger(__name__)
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
    url = os.getenv("AI_SERVICE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("AI_SERVICE_URL must be configured in the environment")
    return url


AI_SERVICE_URL = resolve_ai_service_url()
MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "svm": "Support Vector Machine (SVM)",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
}

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "diabetes_db")
mongo_status = "not_configured"


def init_mongo_client():
    global mongo_status

    if not MONGODB_URI:
        mongo_status = "not_configured"
        print("MongoDB URI không được cấu hình. Lịch sử dự đoán sẽ không được lưu.")
        return None

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        mongo_status = "connected"
        print("MongoDB kết nối thành công. Lịch sử dự đoán sẽ được lưu.")
        return client
    except Exception as exc:
        mongo_status = "unavailable"
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
    if mongo_client is not None:
        try:
            mongo_client.admin.command("ping")
            mongo_status = "connected"
        except Exception:
            mongo_status = "unavailable"

    return {
        "status": "ok",
        "ai_service_url": AI_SERVICE_URL,
        "mongodb": mongo_status,
        "mongodb_database": MONGODB_DATABASE,
    }


@app.post("/api/v1/predict-disease")
def proxy_to_ai_service(data: PatientInput):
    payload = data.model_dump()
    try:
        config_response = requests.get(f"{AI_SERVICE_URL}/model-config", timeout=10)
        if config_response.status_code != 200:
            raise HTTPException(status_code=config_response.status_code, detail=config_response.text)
        config = config_response.json()
        model_keys = config.get("model_keys", list(MODEL_LABELS))
        if set(model_keys) != set(MODEL_LABELS):
            raise HTTPException(status_code=502, detail="AI Service config must include all four models")

        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(
                lambda model_key: requests.post(
                    f"{AI_SERVICE_URL}/predict/{model_key}", json=payload, timeout=20
                ),
                model_keys,
            ))

        results = []
        for model_key, response in zip(model_keys, responses):
            if response.status_code != 200:
                logger.error("AI service returned %s for model %s: %s", response.status_code, model_key, response.text)
                raise HTTPException(status_code=502, detail=f"AI Service prediction failed for {model_key} (HTTP {response.status_code})")
            prediction_result = response.json()
            prediction = int(prediction_result["prediction"])
            metrics = config["models"][model_key]
            threshold = float(metrics["threshold"])
            results.append({
                "modelName": MODEL_LABELS[model_key],
                "prediction": prediction,
                "probability": round(float(prediction_result["probability"]) * 100, 1),
                "threshold": round(threshold * 100, 1),
                "statusText": "Nguy cơ tiểu đường" if prediction == 1 else "Bình thường",
                "description": "Prediction threshold selected by cross-validation on the training split.",
                "cvMetrics": metrics["cv_metrics"],
                "testMetrics": metrics["test_metrics"],
            })

        recommended_model = config["model_name"]
        now_utc = datetime.now(timezone.utc)
        response_payload = {
            "id": f"pred-{now_utc.strftime('%Y%m%d%H%M%S')}",
            "createdAt": now_utc.strftime("%d/%m/%Y %H:%M:%S"),
            "inputData": payload,
            "results": results,
            "recommendedModel": recommended_model,
            "targetSensitivity": config["target_sensitivity"],
            "cvSampleCount": config["cv_sample_count"],
            "testSampleCount": config["test_sample_count"],
        }
        if mongo_client is not None:
            database = mongo_client[MONGODB_DATABASE]
            database["prediction_history"].insert_one({
                "request_id": response_payload["id"], "created_at": now_utc,
                "input_data": payload, "results": results,
                "recommended_model": recommended_model,
                "target_sensitivity": config["target_sensitivity"],
                "model_thresholds": {result["modelName"]: result["threshold"] for result in results},
            })
        return response_payload
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Cannot connect to AI service at {AI_SERVICE_URL}: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction processing failed: {exc}")
