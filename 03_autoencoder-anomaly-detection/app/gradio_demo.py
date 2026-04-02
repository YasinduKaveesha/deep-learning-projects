"""Gradio demo — upload image, view reconstruction + heatmap + anomaly score."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from app.model import AnomalyDetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best_autoencoder.pt"
THRESHOLD = 0.000470
DEVICE = "cpu"
EXAMPLES_DIR = Path(__file__).parent / "examples"

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
detector = AnomalyDetector(MODEL_PATH, THRESHOLD, DEVICE)


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _get_reconstruction(det: AnomalyDetector, pil_image: Image.Image) -> Image.Image:
    """Get the reconstructed image for display."""
    import torch

    from app.model import _TRANSFORM

    x = _TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(det.device)
    with torch.no_grad():
        x_hat, _ = det._model(x)
    # x_hat is in [0, 1] — convert to PIL
    recon_np = (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(recon_np)


def predict(image: Image.Image):
    """Run inference and return outputs for the Gradio interface."""
    if image is None:
        return None, None, "Upload an image to get started."

    result = detector.predict(image)

    # Reconstruction
    recon_image = _get_reconstruction(detector, image)

    # Heatmap overlay from base64
    heatmap_image = _b64_to_pil(result["heatmap_base64"])

    # Summary text
    verdict = "ANOMALY DETECTED" if result["is_anomaly"] else "NORMAL"
    summary = (
        f"## {verdict}\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Reconstruction Error | {result['recon_error']:.6f} |\n"
        f"| Threshold (p95) | {result['threshold_used']:.6f} |\n"
        f"| Confidence | {result['confidence']:.2%} |\n"
        f"| Inference Time | {result['inference_time_ms']:.1f} ms |\n"
        f"| Model | {result['model_version']} |\n"
    )

    return recon_image, heatmap_image, summary


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
examples = sorted(EXAMPLES_DIR.glob("*.png")) if EXAMPLES_DIR.exists() else []

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Reconstruction"),
        gr.Image(type="pil", label="Error Heatmap"),
        gr.Markdown(label="Results"),
    ],
    title="Autoencoder Anomaly Detection",
    description=(
        "Upload a hazelnut image to detect surface defects. "
        "The convolutional autoencoder was trained on defect-free images only. "
        "High reconstruction error = anomaly."
    ),
    examples=[[str(p)] for p in examples] if examples else None,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
