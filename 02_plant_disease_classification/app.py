"""
Plant Disease Classifier — HuggingFace Spaces entry point.

Model  : ResNet50 fine-tuned on PlantVillage
Classes: 38
Test   : 99.66% accuracy, 0.9952 macro-F1

Model weights are downloaded from HuggingFace Hub on first run.
Replace YOUR_HF_USERNAME below before deploying to Spaces.
"""

from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

# ── configuration ──────────────────────────────────────────────────────────
# ↓ Replace with your actual HuggingFace username before deploying
HF_REPO        = "YOUR_HF_USERNAME/plant-disease-resnet50"
HF_FILENAME    = "resnet50_best.pth"
NUM_CLASSES    = 38

BASE_DIR        = Path(__file__).resolve().parent
LOCAL_CHECKPOINT = BASE_DIR / "models" / "resnet50_best.pth"
TRAIN_CSV       = BASE_DIR / "data" / "processed" / "train.csv"


# ── preprocessing — must match training exactly ────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── model loading ──────────────────────────────────────────────────────────
def _get_checkpoint() -> str:
    """Return local checkpoint path if it exists, otherwise download from Hub."""
    if LOCAL_CHECKPOINT.exists():
        print(f"[model] using local checkpoint: {LOCAL_CHECKPOINT}")
        return str(LOCAL_CHECKPOINT)
    print(f"[model] downloading from HuggingFace Hub: {HF_REPO}/{HF_FILENAME}")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def _build_model(checkpoint: str, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


# ── startup: load model + labels once ─────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] device = {DEVICE}")

_checkpoint   = _get_checkpoint()
MODEL         = _build_model(_checkpoint, NUM_CLASSES, DEVICE)
print(f"[startup] model ready")

_train_df     = pd.read_csv(TRAIN_CSV)
_class_names  = sorted(_train_df["label"].unique())
IDX_TO_CLASS  = {i: name for i, name in enumerate(_class_names)}
print(f"[startup] {len(_class_names)} classes loaded")


# ── label formatter ────────────────────────────────────────────────────────
def _fmt(class_name: str) -> str:
    """
    Convert PlantVillage class names to readable strings.

    Examples:
      Tomato___Late_blight              → Tomato — Late Blight
      Corn_(maize)___Northern_Leaf_Blight → Corn (Maize) — Northern Leaf Blight
      Pepper,_bell___healthy            → Pepper, Bell — Healthy
    """
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        plant   = plant.replace("_", " ").strip()
        disease = disease.replace("_", " ").strip().title()
        return f"{plant} — {disease}"
    return class_name.replace("_", " ").title()


# ── inference ──────────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return {}, pd.DataFrame(columns=["Class", "Confidence"])

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)

    probs          = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(3)

    top3 = [
        (IDX_TO_CLASS[i.item()], p.item())
        for i, p in zip(top_idx, top_probs)
    ]

    # gr.Label expects {label: confidence} — highest confidence shown first
    label_dict = {_fmt(name): round(conf, 4) for name, conf in top3}

    # Detailed table
    df = pd.DataFrame(
        [{"Class": _fmt(name), "Confidence": f"{conf:.2%}"} for name, conf in top3]
    )

    return label_dict, df


# ── example images ─────────────────────────────────────────────────────────
# HOW TO ADD EXAMPLE IMAGES:
# 1. Download PlantVillage from:
#    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# 2. Place dataset in data/raw/  (folder structure: data/raw/<ClassName>/<image>.jpg)
# 3. Run this script to copy 10 example images:
#    python scripts/copy_examples.py
# 4. Uncomment the  examples=EXAMPLES  line in gr.Interface below
#
_examples_dir = BASE_DIR / "examples"
EXAMPLES = (
    sorted([[str(p)] for p in sorted(_examples_dir.glob("*.jpg"))])
    if _examples_dir.exists() and any(_examples_dir.glob("*.jpg"))
    else None
)

# ── interface ──────────────────────────────────────────────────────────────
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload a leaf image",
    ),
    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Top Predictions",
        ),
        gr.Dataframe(
            headers=["Class", "Confidence"],
            label="Full Top-3 Breakdown",
            row_count=3,
        ),
    ],
    title="Plant Disease Classifier",
    description=(
        "**ResNet50** fine-tuned on the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) "
        "(54,305 images · 38 classes · 36× class imbalance).\n\n"
        "**Test accuracy: 99.66%** · Macro-F1: 0.9952\n\n"
        "Upload a close-up photo of a plant leaf for an instant disease diagnosis. "
        "Works best with clear, well-lit leaf images similar to the PlantVillage style."
    ),
    # examples=EXAMPLES,  # ← Uncomment after running: python scripts/copy_examples.py
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch(theme=gr.themes.Soft())
