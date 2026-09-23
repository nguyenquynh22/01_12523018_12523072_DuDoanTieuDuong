from fastapi import APIRouter, HTTPException
import requests
import os

router = APIRouter()
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8001")

@router.post("/predict/{model_name}")
def proxy_predict(model_name: str, data: dict):
    try:
        response = requests.post(f"{AI_SERVICE_URL}/predict/{model_name}", json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Lỗi từ AI Service")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))