"""Integration tests for the FastAPI anomaly detection service."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers lifespan (model load) once per session."""
    with TestClient(app) as c:
        yield c


def _white_image_b64(width: int = 224, height: int = 224) -> str:
    """Return a base64-encoded PNG of a solid white RGB image."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health_200(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "ConvAutoencoder-v1"
    assert isinstance(body["threshold"], float)
    assert body["device"] in ("cpu", "cuda")


def test_predict_white_image_valid_schema(client: TestClient):
    payload = {"image_base64": _white_image_b64()}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()

    # All required fields present
    assert "is_anomaly" in body
    assert "recon_error" in body
    assert "threshold_used" in body
    assert "confidence" in body
    assert "heatmap_base64" in body
    assert "model_version" in body
    assert "inference_time_ms" in body

    # Type checks
    assert isinstance(body["is_anomaly"], bool)
    assert isinstance(body["recon_error"], float)
    assert body["recon_error"] > 0
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["heatmap_base64"], str)
    assert len(body["heatmap_base64"]) > 0
    assert body["model_version"] == "ConvAutoencoder-v1"
    assert body["inference_time_ms"] > 0


def test_predict_bad_input_returns_400(client: TestClient):
    payload = {"image_base64": "not-valid-base64!!!"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
