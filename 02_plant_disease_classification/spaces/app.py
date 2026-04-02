"""Plant Disease Classifier — HuggingFace Spaces entry point.

Self-contained Gradio demo. Model weights are downloaded from HuggingFace Hub
on first run. Class names are inlined so no CSV dependency is needed.
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

# -- Configuration ------------------------------------------------------------
HF_REPO = "mykkularathne/plant-disease-resnet50"
HF_FILENAME = "resnet50_best.pth"
NUM_CLASSES = 38

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# -- Model loading ------------------------------------------------------------
def _build_model(checkpoint: str, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] device = {DEVICE}")

# Try local checkpoint first, fall back to HF Hub download
_local_ckpt = Path(__file__).parent / "resnet50_best.pth"
if _local_ckpt.exists():
    _checkpoint = str(_local_ckpt)
    print(f"[model] using local checkpoint: {_checkpoint}")
else:
    _checkpoint = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    print(f"[model] downloaded from HuggingFace Hub: {HF_REPO}/{HF_FILENAME}")

MODEL = _build_model(_checkpoint, NUM_CLASSES, DEVICE)
print(f"[startup] model ready — {NUM_CLASSES} classes")


# -- Label formatter ----------------------------------------------------------
def _fmt(class_name: str) -> str:
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        plant = plant.replace("_", " ").strip()
        disease = disease.replace("_", " ").strip().title()
        return f"{plant} — {disease}"
    return class_name.replace("_", " ").title()


# -- Inference ----------------------------------------------------------------
def predict(image: Image.Image):
    if image is None:
        return {}, pd.DataFrame(columns=["Class", "Confidence"])

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)

    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(3)

    top3 = [
        (IDX_TO_CLASS[i.item()], p.item())
        for i, p in zip(top_idx, top_probs)
    ]

    label_dict = {_fmt(name): round(conf, 4) for name, conf in top3}

    df = pd.DataFrame(
        [{"Class": _fmt(name), "Confidence": f"{conf:.2%}"} for name, conf in top3]
    )

    return label_dict, df


# -- Example images -----------------------------------------------------------
_examples_dir = Path(__file__).parent / "examples"
EXAMPLES = (
    sorted([[str(p)] for p in sorted(_examples_dir.glob("*.jpg"))])
    if _examples_dir.exists() and any(_examples_dir.glob("*.jpg"))
    else None
)

# -- Gradio UI ----------------------------------------------------------------
iface = gr.Interface(
    fn=predict,
    theme=gr.themes.Soft(),
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
        "(54,305 images · 38 classes · 36x class imbalance).\n\n"
        "**Test accuracy: 99.66%** · Macro-F1: 0.9952\n\n"
        "Upload a close-up photo of a plant leaf for an instant disease diagnosis. "
        "Works best with clear, well-lit leaf images similar to the PlantVillage style."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
