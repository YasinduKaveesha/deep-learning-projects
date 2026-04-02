"""HuggingFace Spaces entry point — Gradio anomaly detection demo.

This file is self-contained: it inlines the model architecture, preprocessing,
and heatmap logic so that Spaces only needs this file + the checkpoint.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet constants
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

THRESHOLD = 0.000470


# ---------------------------------------------------------------------------
# Model architecture (inlined)
# ---------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)), self.encoder(x)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "best_autoencoder.pt"
device = "cpu"

model = ConvAutoencoder()
state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state = state.get("model_state_dict", state)
model.load_state_dict(state)
model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def make_heatmap(x_in, x_hat):
    """Generate heatmap overlay as PIL Image."""
    mean = _MEAN
    std = _STD
    x_denorm = (x_in[0] * std[0] + mean[0]).clamp(0.0, 1.0)
    err_map = ((x_hat[0] - x_denorm) ** 2).mean(dim=0).cpu().detach()
    err_u8 = (err_map / (err_map.max() + 1e-8)).mul(255).byte().numpy()
    blurred = cv2.GaussianBlur(err_u8, (0, 0), sigmaX=2)
    heatmap_bgr = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    orig_rgb = (x_denorm.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def predict(image):
    if image is None:
        return None, None, "Upload an image to get started."

    t0 = time.perf_counter()
    x = _TRANSFORM(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        x_hat, _ = model(x)

    x_denorm = (x * _STD + _MEAN).clamp(0.0, 1.0)
    recon_error = float(F.mse_loss(x_hat, x_denorm, reduction="none").mean())
    is_anomaly = recon_error >= THRESHOLD
    confidence = min(abs(recon_error - THRESHOLD) / (THRESHOLD + 1e-12), 1.0)
    ms = (time.perf_counter() - t0) * 1000.0

    recon_img = Image.fromarray(
        (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    heatmap_img = make_heatmap(x, x_hat)

    verdict = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    summary = (
        f"## {verdict}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Reconstruction Error | {recon_error:.6f} |\n"
        f"| Threshold (p95) | {THRESHOLD:.6f} |\n"
        f"| Confidence | {confidence:.2%} |\n"
        f"| Inference Time | {ms:.1f} ms |\n"
    )

    return recon_img, heatmap_img, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
examples_dir = Path(__file__).parent / "examples"
examples = sorted(examples_dir.glob("*.png")) if examples_dir.exists() else []

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
        "Trained on MVTec AD hazelnut (normal images only). "
        "High reconstruction error = anomaly. "
        "**PR-AUC: 0.93 | Best F1: 0.65**"
    ),
    examples=[[str(p)] for p in examples] if examples else None,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
