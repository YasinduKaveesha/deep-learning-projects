"""MVTec AD hazelnut dataset loader for anomaly detection."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


class MVTecDataset(Dataset):
    """PyTorch Dataset for the MVTec AD hazelnut category.

    Args:
        root_dir: Path to the dataset root (the directory containing
                  ``data/mvtec/hazelnut/``).
        split:    One of ``"train"``, ``"val"``, or ``"test"``.
        transform: Optional torchvision transform.  Defaults to
                   Resize(256) → CenterCrop(224) → ToTensor() → Normalize.

    Returns (per item):
        (image_tensor, label, filename)
        label: 0 = normal, 1 = anomaly
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root_dir: str | os.PathLike,
        split: str = "train",
        transform=None,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got {split!r}"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform if transform is not None else _DEFAULT_TRANSFORM

        hazelnut_root = self.root_dir / "data" / "mvtec" / "hazelnut"
        self.samples: list[tuple[Path, int]] = []  # (path, label)

        if split in ("train", "val"):
            good_dir = hazelnut_root / "train" / "good"
            all_files = sorted(good_dir.glob("*.png"))
            if not all_files:
                all_files = sorted(good_dir.glob("*.jpg"))

            # Deterministic 80/20 split — sorted order, last 20% = val
            split_idx = int(len(all_files) * 0.8)
            if split == "train":
                selected = all_files[:split_idx]
            else:
                selected = all_files[split_idx:]

            self.samples = [(p, 0) for p in selected]

        else:  # test
            test_dir = hazelnut_root / "test"
            for category_dir in sorted(test_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                label = 0 if category_dir.name == "good" else 1
                for ext in ("*.png", "*.jpg"):
                    for img_path in sorted(category_dir.glob(ext)):
                        self.samples.append((img_path, label))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, img_path.name

    def __repr__(self) -> str:
        normal = sum(1 for _, lbl in self.samples if lbl == 0)
        anomaly = sum(1 for _, lbl in self.samples if lbl == 1)
        return (
            f"MVTecDataset("
            f"split={self.split}, "
            f"total={len(self.samples)}, "
            f"normal={normal}, "
            f"anomaly={anomaly})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root_dir: str | os.PathLike,
    batch_size: int = 16,
    num_workers: int = 2,
    transform=None,
) -> dict[str, DataLoader]:
    """Create train / val / test DataLoaders for the hazelnut category.

    Args:
        root_dir:    Path to the dataset root (same as ``MVTecDataset``).
        batch_size:  Mini-batch size for all loaders.
        num_workers: Number of worker processes for data loading.
        transform:   Optional shared transform; defaults to the ImageNet preset.

    Returns:
        ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``
    """
    loaders: dict[str, DataLoader] = {}
    for split in MVTecDataset.VALID_SPLITS:
        dataset = MVTecDataset(root_dir, split=split, transform=transform)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
