"""AeroVision LK -- Gradio Demo for HuggingFace Spaces.

YOLOv8 + SAHI aerial vehicle detection on VisDrone imagery.
Compare Standard YOLO (640px) vs SAHI (512px tiles) side by side.
"""

import time
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# -- Config ------------------------------------------------------------------
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "three_wheeler", "bus", "motor",
]

SLICE_SIZE = 512
OVERLAP_RATIO = 0.1
CONF_THRESHOLD = 0.25
IMGSZ = 640

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
ONNX_WEIGHTS = WEIGHTS_DIR / "yolov8s_int8.onnx"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Per-class colors (BGR for OpenCV)
CLASS_COLORS = [
    (220, 20, 60),    # pedestrian - crimson
    (255, 105, 180),  # people - hot pink
    (0, 191, 255),    # bicycle - deep sky blue
    (50, 205, 50),    # car - lime green
    (255, 165, 0),    # van - orange
    (138, 43, 226),   # truck - blue violet
    (255, 255, 0),    # three_wheeler - yellow
    (0, 128, 128),    # bus - teal
    (255, 69, 0),     # motor - orange red
]

# -- Load model once at startup ----------------------------------------------
import torch

_orig_cuda = torch.cuda.is_available
torch.cuda.is_available = lambda: False

sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(ONNX_WEIGHTS),
    confidence_threshold=CONF_THRESHOLD,
    device="cpu",
)

yolo_model = YOLO(str(ONNX_WEIGHTS), task="detect")

torch.cuda.is_available = _orig_cuda

print(f"Models loaded: {ONNX_WEIGHTS.name}")


# -- Helpers -----------------------------------------------------------------
def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes with class labels on image."""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid = det["class_id"]
        color = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (200, 200, 200)
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def run_sahi(img_path: str) -> tuple[list[dict], float]:
    """Run SAHI tiled inference."""
    t0 = time.perf_counter()
    result = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
        verbose=0,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    detections = []
    for obj in result.object_prediction_list:
        cid = int(obj.category.id)
        detections.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}",
            "confidence": round(float(obj.score.value), 4),
            "bbox": [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy],
        })
    return detections, elapsed


def run_standard(img_path: str) -> tuple[list[dict], float]:
    """Run standard YOLO inference (single 640px pass)."""
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=img_path, imgsz=IMGSZ, conf=CONF_THRESHOLD,
        verbose=False, device="cpu",
    )[0]
    elapsed = (time.perf_counter() - t0) * 1000

    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.zeros((0, 4))
    confs = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.zeros(0)
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) else np.zeros(0, dtype=int)

    detections = []
    for box, cid, conf in zip(boxes, cls_ids, confs):
        detections.append({
            "class_id": int(cid),
            "class_name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}",
            "confidence": round(float(conf), 4),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        })
    return detections, elapsed


def build_summary(detections: list[dict], elapsed_ms: float, mode: str) -> tuple[str, list]:
    """Build summary text and per-class table."""
    counts = Counter(d["class_name"] for d in detections)
    summary = (
        f"**Mode:** {mode}  |  "
        f"**Detections:** {len(detections)}  |  "
        f"**Inference:** {elapsed_ms:.0f} ms"
    )
    table = [[name, counts.get(name, 0)] for name in CLASS_NAMES]
    table.sort(key=lambda r: r[1], reverse=True)
    return summary, table


# -- Main inference function -------------------------------------------------
def detect(image: Image.Image, mode: str):
    """Run detection and return results for Gradio."""
    if image is None:
        return None, None, "Upload an image to start.", []

    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cv2.imwrite(tmp_path, img_bgr)

    try:
        if mode == "SAHI (recommended)":
            detections, elapsed_ms = run_sahi(tmp_path)
        else:
            detections, elapsed_ms = run_standard(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    annotated_bgr = draw_boxes(img_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    summary, table = build_summary(detections, elapsed_ms, mode)

    return image, annotated_pil, summary, table


# -- Gradio UI ---------------------------------------------------------------
examples = [
    [str(EXAMPLES_DIR / "example_1.jpg"), "SAHI (recommended)"],
    [str(EXAMPLES_DIR / "example_2.jpg"), "SAHI (recommended)"],
    [str(EXAMPLES_DIR / "example_3.jpg"), "Standard YOLO"],
]

with gr.Blocks(title="AeroVision LK") as demo:
    gr.Markdown(
        "# AeroVision LK -- Aerial Vehicle Detection\n"
        "YOLOv8s + SAHI on VisDrone imagery. "
        "Compare **Standard YOLO** (single 640px pass) vs "
        "**SAHI** (512px tiled inference) for small-object detection.\n\n"
        "Model: ONNX INT8 (11 MB) | 9 classes | CPU inference"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            mode_toggle = gr.Radio(
                choices=["SAHI (recommended)", "Standard YOLO"],
                value="SAHI (recommended)",
                label="Inference Mode",
            )
            run_btn = gr.Button("Run Detection", variant="primary")

    with gr.Row():
        original_out = gr.Image(label="Original", type="pil")
        annotated_out = gr.Image(label="Detections", type="pil")

    summary_text = gr.Markdown(label="Summary")

    class_table = gr.Dataframe(
        headers=["Class", "Count"],
        label="Detections per Class",
        interactive=False,
    )

    run_btn.click(
        fn=detect,
        inputs=[input_image, mode_toggle],
        outputs=[original_out, annotated_out, summary_text, class_table],
    )

    gr.Examples(
        examples=examples,
        inputs=[input_image, mode_toggle],
        outputs=[original_out, annotated_out, summary_text, class_table],
        fn=detect,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
