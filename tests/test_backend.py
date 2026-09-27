import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, "app/backend")

from app.main import app


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class BackendFlowTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_documentation_route_handles_trailing_dot_and_favicon(self):
        docs_response = self.client.get("/docs.", follow_redirects=False)
        self.assertIn(docs_response.status_code, (307, 308, 200))
        self.assertTrue(docs_response.headers.get("location", "").endswith("/docs") or docs_response.status_code == 200)

        favicon_response = self.client.get("/favicon.ico")
        self.assertIn(favicon_response.status_code, (200, 204))

    def test_predict_endpoint_returns_combined_results(self):
        payload = {
            "pregnancies": 6,
            "glucose": 148,
            "bloodPressure": 72,
            "skinThickness": 35,
            "insulin": 0,
           "bmi": 33.6,
            "diabetesPedigreeFunction": 0.627,
            "age": 50,
        }

        fake_responses = {
            "logistic": {"model_used": "logistic", "prediction": 1, "probability": 0.82},
            "svm": {"model_used": "svm", "prediction": 1, "probability": 0.85},
            "naive_bayes": {"model_used": "naive_bayes", "prediction": 0, "probability": 0.62},
            "random_forest": {"model_used": "random_forest", "prediction": 1, "probability": 0.91},
        }

        def fake_post(url, json, timeout=None):
            model_name = url.rstrip("/").split("/")[-1]
            return FakeResponse(200, fake_responses[model_name])

        with patch("app.main.requests.post", side_effect=fake_post):
            response = self.client.post("/api/v1/predict-disease", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 4)
        self.assertEqual(data["bestModel"], "Random Forest")


if __name__ == "__main__":
    unittest.main()
