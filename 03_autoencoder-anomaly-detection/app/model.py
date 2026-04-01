"""Self-contained inference module — no src/ imports.

All model architecture, preprocessing, and heatmap logic is inlined here
so the Docker container only needs app/ and models/.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

MODEL_VERSION = "ConvAutoencoder-v1"


# ---------------------------------------------------------------------------
# Inline ConvAutoencoder architecture (mirrors src/model.py exactly)
# ---------------------------------------------------------------------------
def _build_model() -> nn.Module:
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )

    class _AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return _AutoEncoder()


# ---------------------------------------------------------------------------
# Heatmap generation (cv2/numpy/PIL only — no matplotlib)
# ---------------------------------------------------------------------------
def _make_heatmap_b64(
    x_input: torch.Tensor,    # (1, 3, 224, 224) ImageNet-normalised, on device
    x_hat: torch.Tensor,      # (1, 3, 224, 224) in [0,1], on device
    device: str,
) -> str:
    mean = _MEAN.to(device)
    std  = _STD.to(device)

    # Denormalise input to [0, 1]
    x_denorm = (x_input[0] * std[0] + mean[0]).clamp(0.0, 1.0)  # (3, 224, 224)

    # Per-pixel MSE averaged over channels → (224, 224)
    err_map = ((x_hat[0] - x_denorm) ** 2).mean(dim=0).cpu().detach()

    # Normalise to [0, 255] uint8
    err_u8 = (err_map / (err_map.max() + 1e-8)).mul(255).byte().numpy()

    # Gaussian blur + JET colormap
    blurred = cv2.GaussianBlur(err_u8, (0, 0), sigmaX=2)
    heatmap_bgr = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    # Build BGR original
    orig_rgb = (x_denorm.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # Weighted overlay: 60% original + 40% heatmap
    overlay_bgr = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Encode to PNG → base64
    pil_img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.000470,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.device = device

        model = _build_model()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._model = model

    def predict(self, image: Image.Image) -> dict:
        """Run full inference pipeline on a PIL Image.

        Returns a dict matching PredictResponse fields.
        """
        t0 = time.perf_counter()

        # Preprocess
        x = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)  # (1,3,224,224)

        # Forward pass
        with torch.no_grad():
            x_hat, _ = self._model(x)

        # Reconstruction error (denormalise first, same as src/model.py)
        mean = _MEAN.to(self.device)
        std  = _STD.to(self.device)
        x_denorm = (x * std + mean).clamp(0.0, 1.0)
        recon_error = float(F.mse_loss(x_hat, x_denorm, reduction="none").mean())

        is_anomaly = recon_error >= self.threshold
        confidence = min(abs(recon_error - self.threshold) / (self.threshold + 1e-12), 1.0)

        heatmap_b64 = _make_heatmap_b64(x, x_hat, self.device)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "is_anomaly":        bool(is_anomaly),
            "recon_error":       recon_error,
            "threshold_used":    self.threshold,
            "confidence":        float(confidence),
            "heatmap_base64":    heatmap_b64,
            "model_version":     MODEL_VERSION,
            "inference_time_ms": inference_time_ms,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_detector: AnomalyDetector | None = None


def load_detector(
    model_path: str | Path,
    threshold: float,
    device: str,
) -> None:
    global _detector
    _detector = AnomalyDetector(model_path, threshold, device)


def get_detector() -> AnomalyDetector:
    if _detector is None:
        raise RuntimeError("Model not loaded. Call load_detector() first.")
    return _detector
