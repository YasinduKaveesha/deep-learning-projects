"""Training pipeline for ConvAutoencoder on MVTec AD hazelnut."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import ConvAutoencoder

# ---------------------------------------------------------------------------
# Epoch-level helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ConvAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Run one training epoch.

    Args:
        model:     The autoencoder (will be set to train mode).
        loader:    DataLoader yielding (image, label, filename) 3-tuples.
        optimizer: Optimizer instance.
        device:    Target device.

    Returns:
        Average reconstruction loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for x, _, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        loss = ConvAutoencoder.reconstruction_error(x, x_hat).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model: ConvAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> float:
    """Run one validation pass (no gradient updates).

    Args:
        model:  The autoencoder (will be set to eval mode).
        loader: DataLoader yielding (image, label, filename) 3-tuples.
        device: Target device.

    Returns:
        Average reconstruction loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = ConvAutoencoder.reconstruction_error(x, x_hat).mean()
            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """Train ConvAutoencoder with MLflow tracking and early stopping.

    Args:
        config: Dictionary with keys:
            root_dir   (str)  — project root passed to get_dataloaders
            batch_size (int)  — mini-batch size
            lr         (float)— Adam learning rate
            epochs     (int)  — maximum training epochs
            device     (str)  — "cuda" or "cpu"
            patience   (int)  — early-stopping patience on val_loss
    """
    root_dir   = config["root_dir"]
    batch_size = config["batch_size"]
    lr         = config["lr"]
    epochs     = config["epochs"]
    device     = config["device"]
    patience   = config["patience"]

    # Paths
    project_root  = Path(root_dir)
    models_dir    = project_root / "models"
    figures_dir   = project_root / "reports" / "figures"
    checkpoint    = models_dir / "best_autoencoder.pt"
    curves_path   = figures_dir / "training_curves.png"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Data
    loaders = get_dataloaders(root_dir, batch_size=batch_size)

    # Model + optimiser
    model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # MLflow
    mlflow.set_experiment("autoencoder-anomaly-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "lr":           lr,
            "batch_size":   batch_size,
            "epochs":       epochs,
            "patience":     patience,
            "architecture": "ConvAutoencoder",
        })

        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
            val_loss   = validate(model, loaders["val"], device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint)

            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"best={best_val_loss:.6f}"
                + (" *" if improved else "")
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
            patience_counter += 1 if not improved else 0

        # Loss curves
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="train_loss")
        ax.plot(val_losses,   label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Curves — ConvAutoencoder")
        ax.legend()
        plt.tight_layout()
        fig.savefig(curves_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves → {curves_path}")

        # MLflow artifacts
        mlflow.log_artifact(str(checkpoint))
        mlflow.log_artifact(str(curves_path))
        print(f"MLflow run complete. Best val_loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "root_dir":   ".",
        "batch_size": 16,
        "lr":         1e-4,
        "epochs":     100,
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
        "patience":   10,
    }
    train(config)
