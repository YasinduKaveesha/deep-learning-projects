[![CI](https://github.com/YasinduKaveesha/deep-learning-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/YasinduKaveesha/deep-learning-projects/actions/workflows/ci.yml)

# Deep Learning Projects

Three production-grade deep learning projects covering object detection, image classification, and unsupervised anomaly detection. Each project follows a complete ML lifecycle — from exploratory analysis through model training to deployed inference APIs and live demos.

## Projects

| # | Project | Task | Best Metric | Key Tech | Live Demo |
|---|---------|------|-------------|----------|-----------|
| 01 | [AeroVision LK](01_aerovision_lk/) | Aerial vehicle detection | 54.4% mAP@0.5 | YOLOv8 + SAHI, ONNX INT8 | [HF Spaces](https://huggingface.co/spaces/mykkularathne/aerovision-lk) |
| 02 | [Plant Disease Classification](02_plant_disease_classification/) | Image classification (38 classes) | 99.66% accuracy | ResNet50, EfficientNet-B0, Grad-CAM | [HF Spaces](https://huggingface.co/spaces/mykkularathne/plant-disease-classification) |
| 03 | [Autoencoder Anomaly Detection](03_autoencoder-anomaly-detection/) | Unsupervised defect detection | 0.93 PR-AUC | ConvAutoencoder, MVTec AD | [HF Spaces](https://huggingface.co/spaces/mykkularathne/autoencoder-anomaly-detection) |

---

## Highlights

**01 — AeroVision LK**
YOLOv8s + SAHI on VisDrone2019-DET. Standard YOLO misses 78.7% of annotations under 50px after resize — SAHI tiles the image into 512px slices and runs detection at full resolution. Result: +10.4pp mAP improvement. INT8 quantization reduces model size 3.9x (21.5 MB to 11.0 MB) while preserving 99.3% accuracy.

**02 — Plant Disease Classification**
Two-stage transfer learning (frozen head, then full fine-tune) on PlantVillage. ResNet50 reaches 99.66% test accuracy with only 28 errors out of 8,146 images. Grad-CAM confirms the model attends to lesion tissue, not background shortcuts. All remaining errors involve genuinely ambiguous class boundaries.

**03 — Autoencoder Anomaly Detection**
Convolutional autoencoder (778K params) trained on normal-only MVTec AD hazelnut images. Anomalies are flagged when reconstruction error exceeds a p95-calibrated threshold. PR-AUC of 0.93 with 97%+ precision — the model rarely raises false alarms.

---

## Engineering Stack

Every project ships with:

| Layer | Implementation |
|-------|---------------|
| Inference API | FastAPI with Pydantic validation |
| Interactive Demo | Gradio on HuggingFace Spaces |
| Containerization | Docker (CPU-optimized builds) |
| CI/CD | GitHub Actions — ruff lint + pytest |
| Research | Jupyter notebooks — EDA, training, evaluation, error analysis |

---

## Repository Structure

```
deep-learning-projects/
├── 01_aerovision_lk/                     # Object detection
│   ├── app/                              # FastAPI + Gradio
│   ├── research/                         # 5 notebooks
│   ├── spaces/                           # HuggingFace Spaces deployment
│   ├── tests/                            # API unit tests
│   └── weights/                          # ONNX + PT model weights
│
├── 02_plant_disease_classification/      # Image classification
│   ├── api/                              # FastAPI
│   ├── notebooks/                        # 4 notebooks
│   ├── spaces/                           # HuggingFace Spaces deployment
│   ├── models/                           # Trained checkpoints
│   └── reports/                          # Figures + metrics
│
├── 03_autoencoder-anomaly-detection/     # Anomaly detection
│   ├── app/                              # FastAPI + Gradio
│   ├── notebooks/                        # 4 notebooks
│   ├── spaces/                           # HuggingFace Spaces deployment
│   ├── src/                              # Training + evaluation modules
│   └── reports/                          # Figures + metrics
│
└── README.md
```

## Quick Start

```bash
# Pick any project
cd 01_aerovision_lk

# Install and run the API
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Or run the Gradio demo locally
python app/gradio_demo.py
```

Python 3.10+ required. See each project's README for full setup instructions.
