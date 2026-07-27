"""Tests for the FastAPI inference API."""
from pathlib import Path

import pytest

MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@pytest.fixture
def client():
    """Create a test client, skip if model artifacts missing."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Model artifacts not available (run 'dvc repro' first)")
    from app.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


VALID_PAYLOAD = {
    "temperature": 300.0,
    "pressure": 100.0,
    "vibration": 20.0,
    "humidity": 45.0,
    "equipment": "Turbine",
    "location": "Atlanta",
}


class TestHealthEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


class TestPredictEndpoint:
    def test_valid_prediction_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("Faulty", "Healthy")
        assert "failure_probability" in data
        assert data["status"] == "Success"

    def test_prediction_has_valid_probability(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        prob_str = response.json()["failure_probability"]
        prob_value = float(prob_str.strip("%")) / 100
        assert 0.0 <= prob_value <= 1.0

    def test_missing_field_returns_422(self, client):
        incomplete = {"temperature": 300.0, "pressure": 100.0}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_all_equipment_types_accepted(self, client):
        for equip in ["Turbine", "Compressor", "Pump"]:
            payload = {**VALID_PAYLOAD, "equipment": equip}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200

    def test_all_locations_accepted(self, client):
        for loc in ["Atlanta", "Chicago", "San Francisco", "New York", "Houston"]:
            payload = {**VALID_PAYLOAD, "location": loc}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
