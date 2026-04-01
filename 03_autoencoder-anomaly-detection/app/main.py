"""FastAPI application for autoencoder anomaly detection inference."""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException

from app.model import MODEL_VERSION, get_detector, load_detector
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MODEL_PATH = Path(__file__).parent.parent / "models" / "best_autoencoder.pt"
_THRESHOLD  = 0.000470
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_detector(_MODEL_PATH, _THRESHOLD, _DEVICE)
    yield
    # Nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Autoencoder Anomaly Detection",
    description="Unsupervised industrial defect detection via convolutional autoencoder.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and model metadata."""
    try:
        detector = get_detector()
        return HealthResponse(
            status="ok",
            model_version=MODEL_VERSION,
            threshold=detector.threshold,
            device=detector.device,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_version=MODEL_VERSION,
            threshold=_THRESHOLD,
            device=_DEVICE,
        )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify an image as normal or anomalous.

    Accepts a base64-encoded image (PNG, JPEG, or any PIL-supported format).
    Returns reconstruction error, anomaly flag, confidence, and a heatmap overlay.
    """
    # Decode base64 → PIL Image
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    # Run inference
    try:
        detector = get_detector()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = detector.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictResponse(**result)
