# Plant Disease Classification

Deep learning pipeline for classifying 38 plant disease categories from leaf images.
Trained on the PlantVillage dataset using a custom baseline CNN and two pretrained
architectures (ResNet50, EfficientNet-B0) with two-stage transfer learning.

---

## Project Overview

- **Dataset:** PlantVillage — 38 classes (diseases + healthy variants across 14 crop species), ~54,000 images
- **Split:** 70 / 15 / 15 stratified train / val / test — seed 42, fixed across all experiments
- **Best model:** ResNet50 fine-tuned — Test Accuracy: 99.66%, Test Macro-F1: 0.9952
- **Hardware:** NVIDIA RTX 3050 Laptop 6GB VRAM, mixed precision (AMP) throughout

---

## Results Summary

### Ablation Table

| Model                        | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|------------------------------|-------------|-------------|--------------|---------------|
| Baseline CNN                 | 71.38%      | 0.6668      | 71.37%       | 0.6715        |
| ResNet50 (frozen)            | 93.46%      | 0.9214      | —            | —             |
| ResNet50 (fine-tuned)        | 99.69%      | 0.9953      | **99.66%**   | **0.9952**    |
| EfficientNet-B0 (frozen)     | 89.32%      | 0.8757      | —            | —             |
| EfficientNet-B0 (fine-tuned) | 99.62%      | 0.9948      | 99.63%       | 0.9943        |

Test evaluation run on fine-tuned checkpoints only. Frozen rows show val metrics.

### Training Strategy

Each transfer learning model trained in two stages:

- **Stage 1 — Frozen (10 epochs, lr=1e-3):** Backbone frozen, classifier head only
- **Stage 2 — Fine-tune (20 epochs, lr=1e-4, CosineAnnealingLR):** All layers unfrozen

### Error Analysis

- **Total misclassifications:** 28 of 8,146 test images (0.34%)
- **Primary failure mode:** Corn Cercospora leaf spot ↔ Corn Northern Leaf Blight
  (9 of 28 errors, 32%) — both produce visually identical elongated stripe patterns;
  the model cannot distinguish lesion width or colour tone at 7×7 feature map resolution
- **Secondary failures:** Potato Late Blight → Tomato Late Blight (2 errors, no host-plant
  context), Tomato Target Spot → Tomato healthy (2 errors, early-stage lesions),
  Soybean healthy → 3 different healthy classes (3 errors)

### Grad-CAM Findings

- **Layer:** `model.layer4[-1]` — ResNet50 last residual block, 7×7 spatial resolution
- **Correct predictions:** Activation concentrates tightly over lesion tissue (scab spots,
  mildew colonies, blight patches). Background shows near-zero activation.
- **Incorrect predictions:** Diffuse activation spread across the full leaf or on veins
  and edges rather than lesion tissue.
- **Verdict:** The model has learned genuine disease patterns, not background shortcuts.
  Remaining errors reflect fundamental visual similarity limits in the dataset.

---

## Project Structure

```
02_plant_disease_classification/
├── notebooks/
│   ├── 01_eda.ipynb                           # EDA, class distribution, stratified splits
│   ├── 02_baseline_cnn.ipynb                  # Custom 3-block CNN, 10 epochs
│   ├── 03_transfer_learning_experiments.ipynb # ResNet50 + EfficientNet-B0 two-stage training
│   └── 04_error_analysis_gradcam.ipynb        # Error analysis, confusion matrix, Grad-CAM
├── models/
│   ├── resnet50_best.pth                      # Best ResNet50 checkpoint (not tracked in git)
│   ├── efficientnet_b0_best.pth               # Best EfficientNet-B0 checkpoint (not tracked)
│   ├── baseline_cnn_best.pth                  # Best baseline CNN checkpoint (not tracked)
│   └── baseline_cnn_config.json
├── data/
│   └── processed/
│       ├── train.csv                          # 70% stratified split
│       ├── val.csv                            # 15% stratified split
│       └── test.csv                           # 15% stratified split
├── results/
│   ├── baseline/
│   ├── transfer_learning/
│   │   ├── ablation_table.csv
│   │   ├── classification_report_resnet50.txt
│   │   ├── history_resnet50.json
│   │   └── history_efficientnet_b0.json
│   └── error_analysis/
│       ├── inference_results.csv
│       ├── confusion_matrix_test.png
│       ├── top_confused_pairs.png
│       ├── misclassification_gallery.png
│       ├── gradcam_correct.png
│       ├── gradcam_incorrect.png
│       └── gradcam_borderline.png
├── reports/figures/
│   ├── baseline_training_curves.png
│   ├── baseline_confusion_matrix_test.png
│   └── model_comparison.png
├── src/
├── api/
├── configs/
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download PlantVillage from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
and place images under `data/raw/`. Run the EDA notebook to generate stratified CSV splits:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train baseline CNN

```bash
jupyter notebook notebooks/02_baseline_cnn.ipynb
```

Saves best checkpoint to `models/baseline_cnn_best.pth`.

### 4. Run transfer learning experiments

```bash
jupyter notebook notebooks/03_transfer_learning_experiments.ipynb
```

Trains ResNet50 then EfficientNet-B0 sequentially. Saves checkpoints to `models/`.
Expected runtime on RTX 3050 6GB: ~5–7 hours total
(ResNet50: ~3–4 hrs, EfficientNet-B0: ~2–3 hrs).
Run overnight or reduce `epochs_stage2` for a quick test.

### 5. Error analysis and Grad-CAM

```bash
jupyter notebook notebooks/04_error_analysis_gradcam.ipynb
```

Loads `models/resnet50_best.pth`, runs full inference on the test set, generates
confusion matrix, misclassification gallery, and three Grad-CAM galleries.
No retraining — runs in ~3–4 minutes.

### 6. Single-image inference

```python
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import pandas as pd

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
CKPT_PATH   = Path("models/resnet50_best.pth")

# Rebuild class name mapping from the training split
train_df    = pd.read_csv("data/processed/train.csv")
idx_to_class = dict(zip(
    train_df["encoded_label"],
    train_df["label"]
))

# Build and load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
model.eval().to(DEVICE)

# Transforms — must match training exactly
tfms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Run inference
img    = Image.open("your_leaf.jpg").convert("RGB")
tensor = tfms(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs       = F.softmax(model(tensor), dim=1)
    conf, pred  = probs.max(dim=1)

print(f"Prediction : {idx_to_class[pred.item()]}")
print(f"Confidence : {conf.item():.4f}")
```

---

## Key Findings

1. **Transfer learning gap is decisive.** Fine-tuned ResNet50 (Macro-F1: 0.9952)
   outperforms the baseline CNN (0.6715) by +32.4 percentage points. Even a frozen
   ResNet50 head (0.9214) beats the fully-trained baseline by +25pp.

2. **Fine-tuning Stage 2 is worth it.** Stage 2 adds +7.4pp F1 over the frozen
   backbone for ResNet50 (0.9214 → 0.9953), confirming that adapting deep convolutional
   filters to agricultural imagery is necessary to reach near-perfect performance.

3. **ResNet50 vs EfficientNet-B0.** ResNet50 wins by 0.09pp test Macro-F1
   (0.9952 vs 0.9943) with 25.6M vs 5.3M parameters. EfficientNet-B0 is the
   better choice under strict size or latency constraints.

4. **Remaining errors are irreducible at this scale.** All 28 misclassifications
   involve classes with genuinely ambiguous visual boundaries. Grad-CAM confirms the
   model attends to lesion tissue on correct predictions — further improvement would
   require higher-resolution imaging or multi-scale feature fusion, not more training.
