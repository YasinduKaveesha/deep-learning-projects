"""
Plant Disease Classifier — FastAPI inference endpoint.

Endpoints:
  GET  /health   → model status
  POST /predict  → image upload → class + confidence + top-3
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CHECKPOINT  = BASE_DIR / "models" / "resnet50_best.pth"
TRAIN_CSV   = BASE_DIR / "data" / "processed" / "train.csv"
NUM_CLASSES = 38

# ── preprocessing — must match training exactly ────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── model loader ───────────────────────────────────────────────────────────
def _load_model(checkpoint: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


# ── shared app state (populated at startup) ────────────────────────────────
class _State:
    model: nn.Module
    device: torch.device
    idx_to_class: dict[int, str]


_state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ────────────────────────────────────────────────────────────
    _state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] device = {_state.device}")

    _state.model = _load_model(CHECKPOINT, NUM_CLASSES, _state.device)
    print(f"[startup] model loaded from {CHECKPOINT}")

    train_df = pd.read_csv(TRAIN_CSV)
    class_names = sorted(train_df["label"].unique())
    _state.idx_to_class = {i: name for i, name in enumerate(class_names)}
    print(f"[startup] {len(class_names)} classes loaded")

    yield  # app runs here

    # ── shutdown (nothing to release) ──────────────────────────────────────


# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Classifier",
    description="ResNet50 fine-tuned on PlantVillage (38 classes, 99.66% test accuracy)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "ResNet50", "classes": NUM_CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── validate MIME type ─────────────────────────────────────────────────
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()

    # ── decode image ───────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image error: {exc}")

    # ── preprocess + inference ─────────────────────────────────────────────
    tensor = TRANSFORM(image).unsqueeze(0).to(_state.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _state.model(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = probs.topk(3)

    top3 = [
        {
            "class": _state.idx_to_class[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "predicted_class":  top3[0]["class"],
        "confidence":       top3[0]["confidence"],
        "top3":             top3,
        "inference_time_ms": round(inference_ms, 2),
    }
