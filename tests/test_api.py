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


# Typical readings from the training data (means: 71 °C, 36 PSI, 1.6 mm/s)
VALID_PAYLOAD = {
    "temperature": 70.0,
    "pressure": 35.0,
    "vibration": 1.5,
    "humidity": 50.0,
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


class TestPredictionBehaviour:
    def test_typical_readings_are_healthy(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] == "Healthy"
        assert data["warnings"] == []

    def test_high_vibration_is_faulty(self, client):
        payload = {**VALID_PAYLOAD, "vibration": 4.5}
        assert client.post("/predict", json=payload).json()["prediction"] == "Faulty"

    def test_overheating_is_faulty(self, client):
        payload = {**VALID_PAYLOAD, "temperature": 130.0}
        assert client.post("/predict", json=payload).json()["prediction"] == "Faulty"


class TestInputWarnings:
    def test_out_of_range_value_is_flagged(self, client):
        payload = {**VALID_PAYLOAD, "temperature": 300.0}
        warnings = client.post("/predict", json=payload).json()["warnings"]
        assert len(warnings) == 1
        assert "temperature" in warnings[0]

    def test_unknown_equipment_is_flagged(self, client):
        payload = {**VALID_PAYLOAD, "equipment": "Boiler"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert any("Boiler" in w for w in response.json()["warnings"])
