"""Threshold calibration and evaluation for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support

from src.model import ConvAutoencoder


def compute_dynamic_threshold(
    model: ConvAutoencoder,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    k: float = 2.0,
) -> dict:
    """Calibrate anomaly thresholds from validation-set reconstruction errors.

    The validation split contains only normal images, so the error distribution
    characterises in-distribution behaviour.  Four threshold variants are
    returned so callers can compare them with ``build_comparison_table``.

    Args:
        model:      Trained ConvAutoencoder (will be set to eval mode).
        val_loader: DataLoader over the validation split.
        device:     Target device.
        k:          Multiplier stored in the returned dict (informational).

    Returns:
        dict with keys:
            mu              — mean val error
            sigma           — std  val error
            k               — the k argument
            threshold_mu2   — mu + 2*sigma
            threshold_mu3   — mu + 3*sigma
            threshold_p95   — 95th-percentile of val errors
            threshold_fixed — 0.005 (hardcoded baseline)
            all_errors      — list[float] of every per-image val error
    """
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            errors.extend(ConvAutoencoder.reconstruction_error(x, x_hat).tolist())

    errors_np = np.array(errors, dtype=np.float64)
    mu    = float(errors_np.mean())
    sigma = float(errors_np.std())

    return {
        "mu":              mu,
        "sigma":           sigma,
        "k":               k,
        "threshold_mu2":   mu + 2.0 * sigma,
        "threshold_mu3":   mu + 3.0 * sigma,
        "threshold_p95":   float(np.percentile(errors_np, 95)),
        "threshold_fixed": 0.005,
        "all_errors":      errors,
    }


def evaluate_threshold(
    errors: list | np.ndarray,
    labels: list | np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate a single threshold against ground-truth labels.

    Args:
        errors:    Per-image reconstruction MSE scores (any length-N sequence).
        labels:    Ground-truth binary labels (0=normal, 1=anomaly).
        threshold: Decision boundary — images with error >= threshold → anomaly.

    Returns:
        dict with keys: precision, recall, f1, pr_auc, threshold_value.
        pr_auc is computed from the full score curve (threshold-independent).
    """
    errors_np = np.asarray(errors, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int32)

    preds = (errors_np >= threshold).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average="binary", zero_division=0
    )

    # PR-AUC uses raw scores so it is independent of the chosen threshold
    prec_curve, rec_curve, _ = precision_recall_curve(labels_np, errors_np)
    pr_auc = float(auc(rec_curve, prec_curve))

    return {
        "precision":       float(precision),
        "recall":          float(recall),
        "f1":              float(f1),
        "pr_auc":          pr_auc,
        "threshold_value": float(threshold),
    }


def build_comparison_table(
    model: ConvAutoencoder,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    threshold_dict: dict,
) -> tuple[pd.DataFrame, list[float], list[int]]:
    """Run the model on all test images and compare threshold variants.

    Args:
        model:          Trained ConvAutoencoder (will be set to eval mode).
        test_loader:    DataLoader over the test split (normal + anomaly).
        device:         Target device.
        threshold_dict: Output of ``compute_dynamic_threshold``.

    Returns:
        (df, test_errors, test_labels) where:
            df           — DataFrame with columns:
                           threshold_type, threshold_value,
                           precision, recall, f1, pr_auc
            test_errors  — list[float] of per-image MSE for every test image
            test_labels  — list[int]   of ground-truth labels (0/1)
    """
    model.eval()
    test_errors: list[float] = []
    test_labels: list[int]   = []

    with torch.no_grad():
        for x, lbl, _ in test_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            test_errors.extend(ConvAutoencoder.reconstruction_error(x, x_hat).tolist())
            test_labels.extend(lbl.tolist())

    variants = {
        "fixed": threshold_dict["threshold_fixed"],
        "mu2":   threshold_dict["threshold_mu2"],
        "mu3":   threshold_dict["threshold_mu3"],
        "p95":   threshold_dict["threshold_p95"],
    }

    rows = []
    for name, tval in variants.items():
        metrics = evaluate_threshold(test_errors, test_labels, tval)
        rows.append({"threshold_type": name, **metrics})

    df = pd.DataFrame(rows, columns=[
        "threshold_type", "threshold_value",
        "precision", "recall", "f1", "pr_auc",
    ])

    return df, test_errors, test_labels
