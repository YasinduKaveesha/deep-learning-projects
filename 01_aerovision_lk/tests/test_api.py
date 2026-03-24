import io
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.schemas import Detection


def _make_test_image() -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture()
def client():
    """Test client with mocked model (no weights needed)."""
    with patch("app.model.ModelManager.load"):
        from app.main import app
        from app.model import model_manager

        model_manager.detection_model = MagicMock()
        model_manager.inference_mode = "onnx"
        model_manager.device = "cpu"

        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["inference_mode"] == "onnx"
        assert data["device"] == "cpu"
        assert "model_version" in data

    def test_health_model_not_loaded(self):
        with patch("app.model.ModelManager.load"):
            from app.main import app
            from app.model import model_manager

            model_manager.detection_model = None
            with TestClient(app) as c:
                resp = c.get("/health")
                assert resp.json()["status"] == "model_not_loaded"


class TestClasses:
    def test_classes_returns_9(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_classes"] == 9
        assert "three_wheeler" in data["classes"]
        assert "car" in data["classes"]


class TestPredict:
    def test_predict_success(self, client):
        mock_detections = [
            Detection(class_name="car", confidence=0.92, bbox=[100, 200, 300, 400]),
        ]
        with patch(
            "app.model.ModelManager.predict",
            return_value=(mock_detections, 150.0),
        ):
            resp = client.post(
                "/predict",
                files={"file": ("test.jpg", _make_test_image(), "image/jpeg")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_detections"] == 1
            assert data["detections"][0]["class_name"] == "car"
            assert data["inference_time_ms"] == 150.0

    def test_predict_empty_file(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_model_not_loaded(self):
        with patch("app.model.ModelManager.load"):
            from app.main import app
            from app.model import model_manager

            model_manager.detection_model = None
            with TestClient(app) as c:
                resp = c.post(
                    "/predict",
                    files={"file": ("test.jpg", _make_test_image(), "image/jpeg")},
                )
                assert resp.status_code == 503
