"""Visualisation utilities for anomaly detection analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

from src.model import ConvAutoencoder

# ImageNet stats — defined locally to avoid coupling to model.py internals
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Threshold variant display config: (dict_key, label, color, linestyle)
_THRESHOLD_STYLES = [
    ("threshold_fixed", "fixed=0.005",  "black",  "--"),
    ("threshold_mu2",   "mu+2\u03c3",   "orange", "-."),
    ("threshold_mu3",   "mu+3\u03c3",   "purple", ":"),
    ("threshold_p95",   "p95",          "brown",  "--"),
]


def plot_reconstruction_errors(
    val_errors: list[float],
    test_normal_errors: list[float],
    test_anomaly_errors: list[float],
    threshold_dict: dict,
    save_path: str | Path,
) -> None:
    """Overlay val-normal, test-normal, and test-anomaly error histograms.

    Vertical lines mark each threshold variant for visual comparison.

    Args:
        val_errors:          Per-image MSE from the validation split (all normal).
        test_normal_errors:  Per-image MSE for normal test images (label=0).
        test_anomaly_errors: Per-image MSE for anomalous test images (label=1).
        threshold_dict:      Output of ``compute_dynamic_threshold``.
        save_path:           Destination path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(val_errors,          bins=40, density=True, alpha=0.5,
            color="green", label="Val normal")
    ax.hist(test_normal_errors,  bins=40, density=True, alpha=0.5,
            color="blue",  label="Test normal")
    ax.hist(test_anomaly_errors, bins=40, density=True, alpha=0.5,
            color="red",   label="Test anomaly")

    for key, label, color, ls in _THRESHOLD_STYLES:
        ax.axvline(threshold_dict[key], color=color, linestyle=ls,
                   linewidth=1.5, label=label)

    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution by Split & Threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {Path(save_path).resolve()}")


def plot_error_heatmap(
    model: ConvAutoencoder,
    image_tensor: torch.Tensor,
    device: torch.device | str,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Generate a per-pixel error heatmap overlaid on the original image.

    Args:
        model:        Trained ConvAutoencoder (set to eval mode internally).
        image_tensor: Single image tensor of shape (3, 224, 224),
                      ImageNet-normalised.
        device:       Target device.
        save_path:    If provided, the BGR overlay is written to this path.

    Returns:
        RGB uint8 numpy array of shape (224, 224, 3) — suitable for imshow.
    """
    model.eval()
    mean = _MEAN.to(device)
    std  = _STD.to(device)

    x = image_tensor.unsqueeze(0).to(device)          # (1, 3, 224, 224)
    with torch.no_grad():
        x_hat, _ = model(x)                           # (1, 3, 224, 224) in [0,1]

    # Denormalise input to [0,1]
    x_denorm = (x[0] * std + mean).clamp(0.0, 1.0)   # (3, 224, 224)

    # Per-pixel MSE averaged over channels → (224, 224)
    err_map = ((x_hat[0] - x_denorm) ** 2).mean(dim=0).cpu()

    # Normalise to [0, 255] uint8 with zero-division guard
    err_u8 = (err_map / (err_map.max() + 1e-8)).mul(255).byte().numpy()

    # Gaussian blur to smooth the error map
    blurred = cv2.GaussianBlur(err_u8, (0, 0), sigmaX=2)

    # Apply JET colormap (returns BGR)
    heatmap_bgr = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    # Build BGR original from denormalised tensor
    orig_rgb = (x_denorm.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # Weighted overlay
    overlay_bgr = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)

    if save_path is not None:
        cv2.imwrite(str(save_path), overlay_bgr)

    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def plot_heatmap_grid(
    model: ConvAutoencoder,
    test_dataset,
    device: torch.device | str,
    n_normal: int = 4,
    n_anomaly: int = 4,
    save_path: str | Path | None = None,
) -> None:
    """Show original | heatmap pairs for normal and anomalous test images.

    Args:
        model:        Trained ConvAutoencoder.
        test_dataset: MVTecDataset instance with split="test".
        device:       Target device.
        n_normal:     Number of normal images to include.
        n_anomaly:    Number of anomalous images to include.
        save_path:    If provided, figure is saved here.
    """
    normal_idxs  = [i for i, (_, lbl) in enumerate(test_dataset.samples)
                    if lbl == 0][:n_normal]
    anomaly_idxs = [i for i, (_, lbl) in enumerate(test_dataset.samples)
                    if lbl == 1][:n_anomaly]
    all_idxs     = normal_idxs + anomaly_idxs
    n_rows       = len(all_idxs)

    fig, axes = plt.subplots(n_rows, 2, figsize=(8, n_rows * 3))

    mean = _MEAN  # CPU, for display
    std  = _STD

    for row, idx in enumerate(all_idxs):
        img_tensor, lbl, fname = test_dataset[idx]
        tag = "normal" if lbl == 0 else "anomaly"

        # Per-image MSE score
        with torch.no_grad():
            x_in  = img_tensor.unsqueeze(0).to(device)
            x_hat, _ = model(x_in)
        err = ConvAutoencoder.reconstruction_error(x_in, x_hat).item()

        # Denormed original for display
        orig_rgb = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Heatmap overlay
        overlay_rgb = plot_error_heatmap(model, img_tensor, device)

        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f"[{tag}] {Path(fname).stem}  mse={err:.4f}",
                                fontsize=7)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_rgb)
        axes[row, 1].set_title("heatmap", fontsize=7)
        axes[row, 1].axis("off")

    plt.suptitle("Original | Error Heatmap  (top=normal, bottom=anomaly)",
                 fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {Path(save_path).resolve()}")

    plt.show()
    plt.close(fig)


def plot_pr_curve(
    test_errors: list[float],
    test_labels: list[int],
    threshold_dict: dict,
    save_path: str | Path,
) -> None:
    """Plot the Precision-Recall curve with threshold operating points marked.

    Args:
        test_errors:    Per-image MSE scores for all test images.
        test_labels:    Ground-truth labels (0=normal, 1=anomaly).
        threshold_dict: Output of ``compute_dynamic_threshold``.
        save_path:      Destination path for the saved figure.
    """
    errors_np = np.asarray(test_errors, dtype=np.float64)
    labels_np = np.asarray(test_labels, dtype=np.int32)

    prec_curve, rec_curve, thresholds = precision_recall_curve(labels_np, errors_np)
    pr_auc = float(auc(rec_curve, prec_curve))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_curve, prec_curve, color="steelblue", linewidth=2,
            label=f"PR curve (AUC={pr_auc:.3f})")

    # Mark operating points for mu2, mu3, p95
    op_variants = [
        ("threshold_mu2", "mu+2\u03c3", "orange"),
        ("threshold_mu3", "mu+3\u03c3", "purple"),
        ("threshold_p95", "p95",        "brown"),
    ]
    for key, label, color in op_variants:
        tval = threshold_dict[key]
        # Find the index of the closest threshold in the sklearn array
        if len(thresholds) > 0:
            idx = int(np.argmin(np.abs(thresholds - tval)))
            op_rec  = rec_curve[idx]
            op_prec = prec_curve[idx]
            ax.scatter(op_rec, op_prec, color=color, s=80, zorder=5)
            ax.annotate(
                f" {label}\n rec={op_rec:.2f}\n pre={op_prec:.2f}",
                xy=(op_rec, op_prec),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — ConvAutoencoder Anomaly Detection")
    ax.legend(fontsize=9)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {Path(save_path).resolve()}")
