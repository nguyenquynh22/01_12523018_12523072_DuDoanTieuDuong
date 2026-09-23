import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Web Backend API")

# Cấu hình CORS cho Frontend gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Địa chỉ URL của AI-Service (Khi chạy trên Docker, gọi bằng tên service name: 'ai-service')
AI_SERVICE_URL = "http://ai-service:8001/predict"

class PatientInput(BaseModel):
    pregnancies: float
    glucose: float
    bloodPressure: float
    skinThickness: float
    insulin: float
    bmi: float
    diabetesPedigreeFunction: float
    age: float

@app.post("/api/v1/predict-disease")
def proxy_to_ai_service(data: PatientInput):
    try:
        # Backend đóng vai trò client, gọi sang AI-Service qua HTTP POST nội bộ
        response = requests.post(AI_SERVICE_URL, json=data.dict())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="AI Service từ chối xử lý")
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Không thể kết nối đến AI-Service. Kiểm tra lại Docker containers!")