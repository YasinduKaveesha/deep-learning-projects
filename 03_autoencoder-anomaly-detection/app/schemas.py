"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_base64: str  # base64-encoded image bytes (PNG, JPEG, etc.)


class PredictResponse(BaseModel):
    is_anomaly: bool
    recon_error: float        # per-image MSE
    threshold_used: float     # p95 = 0.000470
    confidence: float         # abs(recon_error - threshold) / threshold, clipped [0, 1]
    heatmap_base64: str       # base64-encoded PNG of JET heatmap overlay
    model_version: str        # "ConvAutoencoder-v1"
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str       # "ok" | "degraded"
    model_version: str
    threshold: float
    device: str       # "cpu" or "cuda"
