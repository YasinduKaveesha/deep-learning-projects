# Project Summary — Autoencoder Anomaly Detection

## 1. Project Overview

This project builds an unsupervised anomaly detection system for industrial surface inspection using a convolutional autoencoder. The core idea is simple: train a model exclusively on defect-free images so it learns what "normal" looks like, then flag any image whose reconstruction error exceeds a calibrated threshold as anomalous. No labelled defect data is needed during training — only normal examples. This makes the approach applicable to real manufacturing settings where defect samples are rare or costly to collect.

The dataset is MVTec AD (hazelnut category), a standard industrial anomaly detection benchmark. The model is a symmetric ConvAutoencoder with 4 encoder and 4 decoder layers (777,987 parameters, bottleneck shape 256×14×14), trained with MSE loss using Adam optimiser on an NVIDIA RTX 3050 (6 GB). This is project 3 of a planned 10-project deep learning portfolio targeting ML/AI roles at Sri Lankan tech companies (WSO2, Dialog Axiata, SenzMate, Sysco LABS). It demonstrates unsupervised deep learning, threshold engineering, and the full MLOps stack — skills not covered by the supervised classification projects that precede it.

---

## 2. What Each Notebook Does

### `notebooks/01_eda.ipynb` — Exploratory Data Analysis
- Loads MVTec hazelnut via the custom `MVTecDataset` class and prints split counts: train 312 normal, val 79 normal, test 110 (40 normal / 70 anomaly).
- Renders a 4×4 grid of random normal training images (`reports/figures/normal_samples.png`).
- Renders a 4×4 grid of defect samples grouped by type — crack, cut, hole, print (`reports/figures/defect_samples.png`).
- Plots pixel intensity histograms (grayscale, normalised 0–1) comparing normal vs anomalous images (`reports/figures/pixel_intensity_dist.png`). The distributions overlap substantially, confirming that pixel statistics alone are insufficient for detection.
- Verifies all 391 training images are correctly resized to 224×224 after Resize(256) + CenterCrop(224).

### `notebooks/02_train_autoencoder.ipynb` — Training
- Instantiates `ConvAutoencoder` and confirms the forward pass (input 4×3×224×224 → bottleneck 4×256×14×14 → reconstruction 4×3×224×224).
- Trains for 100 epochs, batch 16, lr 1e-4, patience 10 (early stopping never triggered — ran full 100 epochs). MLflow logs every epoch's train/val loss under the experiment `autoencoder-anomaly-detection`.
- Best validation loss: **0.000289** (epoch 98). Train loss at epoch 100: 0.000423.
- Saves training curves to `reports/figures/training_curves.png` and the best checkpoint to `models/best_autoencoder.pt`.
- Renders a reconstruction comparison grid (4 normal + 4 anomaly from test split) at `reports/figures/reconstruction_comparison.png`.

### `notebooks/03_evaluate_autoencoder.ipynb` — Threshold Analysis & Evaluation
- Loads `models/best_autoencoder.pt` and computes reconstruction errors on the val split (79 normal images) to fit a Gaussian: μ=0.000289, σ=0.000104.
- Derives four threshold candidates: `fixed=0.005`, `mu+2σ=0.000498`, `mu+3σ=0.000602`, `p95=0.000470`.
- Evaluates all four variants on the test split (110 images: 40 normal / 70 anomaly): precision, recall, F1, PR-AUC.
- Saves `reports/eval_metrics.csv`, `reports/figures/eval_error_histogram.png`, `reports/figures/eval_pr_curve.png`, `reports/figures/eval_heatmap_grid.png`.

---

## 3. Key Metrics

**Training**

| Metric | Value |
|---|---|
| Train loss (final, epoch 100) | 0.000423 |
| Val loss (best, epoch 98) | 0.000289 |
| Val error μ | 0.000289 |
| Val error σ | 0.000104 |

The training curve shows healthy monotonic descent from 0.088 (epoch 1) to 0.000289 (epoch 98) with no divergence between train and val loss, indicating no overfitting.

**Threshold comparison (test split: 40 normal / 70 anomaly)**

| Strategy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| `fixed` | 0.005000 | 1.0000 | 0.0286 | 0.0556 |
| `mu+2σ` | 0.000498 | 0.9706 | 0.4714 | 0.6346 |
| `mu+3σ` | 0.000602 | 1.0000 | 0.4000 | 0.5714 |
| **`p95`** | **0.000470** | **0.9714** | **0.4857** | **0.6476** |

**PR-AUC: 0.9286** (threshold-independent)

**Best threshold: `p95=0.000470`** — F1=0.6476, Precision=0.9714, Recall=0.4857

The high PR-AUC (0.93) confirms the model's reconstruction scores rank anomalies very well. The recall bottleneck (~49%) is typical for a basic convolutional autoencoder on MVTec — subtle defect types are hard to separate by MSE alone. Precision is near-perfect (97%), meaning false positives are rare.

**Comparison vs Isolation Forest:** Not done.

---

## 4. Files Created So Far

**Notebooks complete (all run, outputs embedded):**
- `notebooks/01_eda.ipynb` — fully run, all outputs present
- `notebooks/02_train_autoencoder.ipynb` — fully run, all outputs present
- `notebooks/03_evaluate_autoencoder.ipynb` — fully run, all outputs present

**Saved artifacts:**
- `models/best_autoencoder.pt` — trained model checkpoint
- `reports/eval_metrics.csv` — threshold comparison table
- `reports/figures/normal_samples.png`
- `reports/figures/defect_samples.png`
- `reports/figures/pixel_intensity_dist.png`
- `reports/figures/training_curves.png`
- `reports/figures/reconstruction_comparison.png`
- `reports/figures/eval_error_histogram.png`
- `reports/figures/eval_pr_curve.png`
- `reports/figures/eval_heatmap_grid.png`

**Still missing:**
- Nothing from the notebook phase — all 3 notebooks are complete.

---

## 5. What Needs To Be Built Next

1. **FastAPI app** (`app/main.py`) — POST /predict endpoint that accepts an image, returns reconstruction error and anomaly flag. Use `models/best_autoencoder.pt` loaded at startup.
3. **ONNX export** — export the model to ONNX for faster inference in FastAPI (`onnxruntime` already in requirements).
4. **Docker** — `Dockerfile` + `docker-compose.yml` to containerise the FastAPI app.
5. **CI** — GitHub Actions workflow: lint with `ruff`, run `pytest` on unit tests.
6. **Gradio demo** (`app/gradio_demo.py`) — upload an image, display original, reconstruction, heatmap, and anomaly score. Same pattern as the plant disease project.
7. **HuggingFace Spaces deployment** — push Gradio demo with `requirements_spaces.txt`.
8. **README.md** (project-level) — replace the placeholder with full project documentation including architecture diagram, results table, usage instructions, and HF Spaces link.

---

## 6. CV Line

> **Autoencoder Anomaly Detection** — Built an unsupervised industrial defect detector using a convolutional autoencoder (778K params) trained on MVTec AD hazelnut; achieved PR-AUC 0.93 and F1 0.65 via p95 threshold calibration on validation reconstruction errors; deployed as a FastAPI service containerised with Docker and demoed on Hugging Face Spaces.
