"""Convolutional Autoencoder for MVTec AD anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ImageNet stats — same as dataset.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder: 224x224 RGB -> bottleneck (256x14x14) -> 224x224 RGB.

    Encoder downsamples 4x via strided convolutions (224 -> 112 -> 56 -> 28 -> 14).
    Decoder mirrors with ConvTranspose2d.  Final activation is Sigmoid so
    reconstructions are in [0, 1].

    Usage::
        model = ConvAutoencoder()
        x_hat, z = model(x)          # forward
        errors   = model.reconstruction_error(x, x_hat)  # per-image MSE
    """

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            # 3 x 224 x 224 -> 32 x 112 x 112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 x 112 x 112 -> 64 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 56 x 56 -> 128 x 28 x 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 28 x 28 -> 256 x 14 x 14  (bottleneck)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(
            # 256 x 14 x 14 -> 128 x 28 x 28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 28 x 28 -> 64 x 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 56 x 56 -> 32 x 112 x 112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 x 112 x 112 -> 3 x 224 x 224
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to bottleneck representation.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Bottleneck tensor of shape (B, 256, 14, 14).
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode bottleneck to reconstructed image.

        Args:
            z: Bottleneck tensor of shape (B, 256, 14, 14).

        Returns:
            Reconstructed tensor of shape (B, 3, 224, 224) in [0, 1].
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            (reconstruction, bottleneck):
                reconstruction — shape (B, 3, 224, 224), values in [0, 1]
                bottleneck     — shape (B, 256, 14, 14)
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_error(x: Tensor, x_hat: Tensor) -> Tensor:
        """Per-image reconstruction MSE with both tensors in [0, 1].

        The input ``x`` is ImageNet-normalised; it is denormalised to [0, 1]
        before computing the error so the scale matches the Sigmoid output
        of the decoder.

        Args:
            x:     Original input, shape (B, 3, 224, 224), ImageNet-normalised.
            x_hat: Reconstruction, shape (B, 3, 224, 224), values in [0, 1].

        Returns:
            1-D tensor of shape (B,) — per-image MSE.
        """
        mean = _MEAN.to(x.device)
        std  = _STD.to(x.device)
        x_denorm = (x * std + mean).clamp(0.0, 1.0)
        return F.mse_loss(x_hat, x_denorm, reduction="none").mean(dim=[1, 2, 3])

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return a summary dict with parameter counts and tensor shapes."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params":      total,
            "trainable_params":  trainable,
            "input_shape":       (3, 224, 224),
            "bottleneck_shape":  (256, 14, 14),
            "output_shape":      (3, 224, 224),
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        lines = [
            "ConvAutoencoder",
            "=" * 52,
            "Encoder",
            "  Conv2d(  3 ->  32, k=3, s=2, p=1)  224 -> 112",
            "  Conv2d( 32 ->  64, k=3, s=2, p=1)  112 ->  56",
            "  Conv2d( 64 -> 128, k=3, s=2, p=1)   56 ->  28",
            "  Conv2d(128 -> 256, k=3, s=2, p=1)   28 ->  14  [bottleneck]",
            "-" * 52,
            "Decoder",
            "  ConvT(256 -> 128, k=3, s=2, p=1)   14 ->  28",
            "  ConvT(128 ->  64, k=3, s=2, p=1)   28 ->  56",
            "  ConvT( 64 ->  32, k=3, s=2, p=1)   56 -> 112",
            "  ConvT( 32 ->   3, k=3, s=2, p=1)  112 -> 224  + Sigmoid",
            "=" * 52,
            f"  Total params:     {info['total_params']:,}",
            f"  Trainable params: {info['trainable_params']:,}",
            f"  Bottleneck:       {info['bottleneck_shape']}",
        ]
        return "\n".join(lines)
