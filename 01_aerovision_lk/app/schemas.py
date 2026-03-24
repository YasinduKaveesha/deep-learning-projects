from pydantic import BaseModel


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class PredictResponse(BaseModel):
    detections: list[Detection]
    n_detections: int
    inference_time_ms: float
    model_version: str
    inference_mode: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    inference_mode: str
    device: str
