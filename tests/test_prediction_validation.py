import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
AI_MODELS_DIR = ROOT / "ai-models"
sys.path.insert(0, str(AI_MODELS_DIR))

from service.main import app


client = TestClient(app)


def test_ai_service_accepts_partial_payload():
    response = client.post(
        "/predict/logistic",
        json={
            "pregnancies": 2,
            "glucose": 120,
            "bloodPressure": 70,
            "skinThickness": 20,
            "insulin": 80,
            "bmi": 25.5,
            "diabetesPedigreeFunction": 0.5,
            "age": 30,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_used"] == "logistic"
    assert "prediction" in payload
    assert "probability" in payload


def test_ai_service_accepts_blank_strings():
    response = client.post(
        "/predict/svm",
        json={
            "pregnancies": "2",
            "glucose": "120",
            "bloodPressure": "",
            "skinThickness": "20",
            "insulin": "80",
            "bmi": "25.5",
            "diabetesPedigreeFunction": "0.5",
            "age": "30",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_used"] == "svm"
