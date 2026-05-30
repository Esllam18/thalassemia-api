"""
=============================================================
 Thalassemia API — Test Suite
=============================================================
How to run:
    pip install pytest httpx
    pytest test_main.py -v

What we test:
  1. /health endpoint returns correct structure
  2. /predict/manual with valid CBC values
  3. /predict/manual with out-of-range values (should fail 422)
  4. /predict/manual with missing fields (should fail 422)
  5. /predict/file with valid CSV
  6. /predict/file with wrong extension (should fail 415)
  7. /predict/file with missing columns (should fail 422)
  8. /predict/image with wrong extension (should fail 415)

Note: Tests use a MOCK model so you don't need the real .pkl files.
"""

import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Setup: Mock the ML model BEFORE importing main.py
# This way tests work even without the real .pkl files.
# ─────────────────────────────────────────────────────────────────────────────

# Create a fake model that always predicts class 0 (Normal) with 90% confidence
mock_model = MagicMock()
mock_model.predict.return_value = np.array([0])
mock_model.predict_proba.return_value = np.array([[0.90, 0.05, 0.05]])

# Create a fake label encoder
mock_encoder = MagicMock()
mock_encoder.inverse_transform.return_value = ["Normal"]

# Patch joblib.load before importing main
with patch("joblib.load") as mock_load:
    mock_load.side_effect = [mock_model, mock_encoder]
    import main  # noqa: E402

# Apply the mocks to the module globals
main._model = mock_model
main._label_encoder = mock_encoder

# Create test client
client = TestClient(main.app)


# ─────────────────────────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────────────────────────

# Normal CBC values — healthy adult
NORMAL_CBC = {"hgb": 13.5, "mcv": 88.0, "mch": 29.0, "rbc": 4.8}

# Thalassemia trait — low MCV, normal RBC
THAL_CBC = {"hgb": 10.5, "mcv": 65.0, "mch": 20.0, "rbc": 5.2}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    """Health endpoint should always return 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure():
    """Health response must contain all required fields."""
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "encoder_loaded" in data
    assert "version" in data


def test_health_status_is_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. /predict/manual — valid inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_manual_returns_200():
    response = client.post("/predict/manual", json=NORMAL_CBC)
    assert response.status_code == 200


def test_predict_manual_response_has_all_fields():
    """Response must have prediction, thalassemia_score, confidence, etc."""
    response = client.post("/predict/manual", json=NORMAL_CBC)
    data = response.json()
    assert "prediction" in data
    assert "thalassemia_score" in data
    assert "confidence" in data
    assert "recommendation" in data
    assert "explanation" in data
    assert "indicators" in data


def test_predict_manual_indicators_structure():
    """Indicators must contain mentzer_index and shine_lal."""
    response = client.post("/predict/manual", json=NORMAL_CBC)
    indicators = response.json()["indicators"]
    assert "mentzer_index" in indicators
    assert "shine_lal" in indicators


def test_predict_manual_confidence_is_float_between_0_and_1():
    response = client.post("/predict/manual", json=NORMAL_CBC)
    confidence = response.json()["confidence"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_manual_score_is_int_between_1_and_10():
    response = client.post("/predict/manual", json=THAL_CBC)
    score = response.json()["thalassemia_score"]
    assert isinstance(score, int)
    assert 1 <= score <= 10


# ─────────────────────────────────────────────────────────────────────────────
# 3. /predict/manual — invalid inputs (must return 422)
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_manual_negative_hgb_rejected():
    data = {**NORMAL_CBC, "hgb": -1.0}
    response = client.post("/predict/manual", json=data)
    assert response.status_code == 422


def test_predict_manual_zero_rbc_rejected():
    data = {**NORMAL_CBC, "rbc": 0.0}
    response = client.post("/predict/manual", json=data)
    assert response.status_code == 422


def test_predict_manual_hgb_above_max_rejected():
    data = {**NORMAL_CBC, "hgb": 999.0}
    response = client.post("/predict/manual", json=data)
    assert response.status_code == 422


def test_predict_manual_missing_field_rejected():
    """Missing any required field should return 422."""
    data = {"hgb": 13.5, "mcv": 88.0, "mch": 29.0}  # rbc missing
    response = client.post("/predict/manual", json=data)
    assert response.status_code == 422


def test_predict_manual_string_value_rejected():
    """String values should return 422."""
    data = {**NORMAL_CBC, "hgb": "not-a-number"}
    response = client.post("/predict/manual", json=data)
    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 4. MCV override rule
# ─────────────────────────────────────────────────────────────────────────────

def test_high_mcv_overrides_to_normal():
    """
    When MCV >= 81, the result should be overridden to 'Normal'
    even if the model predicts otherwise.
    """
    # Force the mock model to predict class 1 (non-Normal)
    mock_encoder.inverse_transform.return_value = ["Beta Thalassemia Trait"]
    mock_model.predict.return_value = np.array([1])

    high_mcv_data = {**NORMAL_CBC, "mcv": 90.0}
    response = client.post("/predict/manual", json=high_mcv_data)
    assert response.json()["prediction"] == "Normal"

    # Restore mock
    mock_encoder.inverse_transform.return_value = ["Normal"]
    mock_model.predict.return_value = np.array([0])


# ─────────────────────────────────────────────────────────────────────────────
# 5. /predict/file — valid CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_file_valid_csv():
    csv_content = b"Name,HGB,MCV,MCH,RBC\nPatient1,13.5,88.0,29.0,4.8\nPatient2,10.5,65.0,20.0,5.2\n"
    response = client.post(
        "/predict/file",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["patient"] == "Patient1"
    assert "result" in data[0]


def test_predict_file_csv_without_name_column():
    """Name column is optional."""
    csv_content = b"HGB,MCV,MCH,RBC\n13.5,88.0,29.0,4.8\n"
    response = client.post(
        "/predict/file",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 6. /predict/file — invalid inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_file_wrong_extension_rejected():
    response = client.post(
        "/predict/file",
        files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
    )
    assert response.status_code == 415


def test_predict_file_missing_columns_rejected():
    csv_content = b"Name,HGB,MCV\nPatient1,13.5,88.0\n"  # Missing MCH and RBC
    response = client.post(
        "/predict/file",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 422


def test_predict_file_empty_csv_returns_empty_list():
    csv_content = b"HGB,MCV,MCH,RBC\n"  # Header only, no data rows
    response = client.post(
        "/predict/file",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# 7. /predict/image — extension validation
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_image_wrong_extension_rejected():
    response = client.post(
        "/predict/image",
        files={"file": ("report.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")},
    )
    assert response.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# 8. Mentzer Index calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_mentzer_index_calculated_correctly():
    """Mentzer = MCV / RBC. With MCV=65 and RBC=5.2, result = 12.5"""
    response = client.post("/predict/manual", json=THAL_CBC)
    mentzer = response.json()["indicators"]["mentzer_index"]
    expected = round(65.0 / 5.2, 2)
    assert abs(mentzer - expected) < 0.01


def test_shine_lal_calculated_correctly():
    """Shine-Lal = (MCV² × MCH) / 100"""
    response = client.post("/predict/manual", json=THAL_CBC)
    shine = response.json()["indicators"]["shine_lal"]
    expected = round((65.0 ** 2 * 20.0) / 100, 2)
    assert abs(shine - expected) < 0.01
