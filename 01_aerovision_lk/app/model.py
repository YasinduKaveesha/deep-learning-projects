import os
import time
import tempfile
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from app.schemas import Detection

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "three_wheeler", "bus", "motor",
]

# SAHI best config from grid search (notebook 04)
SLICE_SIZE = 512
OVERLAP_RATIO = 0.1
CONF_THRESHOLD = 0.25

MODEL_VERSION = "yolov8s-aerovision-v1"

# Paths relative to project root
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
PYTORCH_WEIGHTS = WEIGHTS_DIR / "yolov8s_baseline.pt"
ONNX_WEIGHTS = WEIGHTS_DIR / "yolov8s_int8.onnx"


class ModelManager:
    def __init__(self):
        self.detection_model = None
        self.inference_mode: str = "onnx"
        self.device: str = "cpu"
        self.model_version: str = MODEL_VERSION

    def load(self) -> None:
        self.inference_mode = os.environ.get("INFERENCE_MODE", "onnx").lower()

        if self.inference_mode == "pytorch":
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_path = str(PYTORCH_WEIGHTS)
        else:
            self.inference_mode = "onnx"
            self.device = "cpu"
            model_path = str(ONNX_WEIGHTS)
            # Monkey-patch: hide CUDA to prevent Ultralytics device query bug
            import torch
            _orig = torch.cuda.is_available
            torch.cuda.is_available = lambda: False

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=CONF_THRESHOLD,
            device=self.device,
        )

        if self.inference_mode == "onnx":
            torch.cuda.is_available = _orig

    def predict(self, image_bytes: bytes) -> tuple[list[Detection], float]:
        """Run SAHI sliced prediction. Returns (detections, elapsed_ms)."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.perf_counter()
            result = get_sliced_prediction(
                tmp_path,
                self.detection_model,
                slice_height=SLICE_SIZE,
                slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP_RATIO,
                overlap_width_ratio=OVERLAP_RATIO,
                verbose=0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        detections = []
        for obj in result.object_prediction_list:
            cid = int(obj.category.id)
            detections.append(Detection(
                class_name=CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}",
                confidence=round(float(obj.score.value), 4),
                bbox=[
                    float(obj.bbox.minx),
                    float(obj.bbox.miny),
                    float(obj.bbox.maxx),
                    float(obj.bbox.maxy),
                ],
            ))

        return detections, elapsed_ms


model_manager = ModelManager()
