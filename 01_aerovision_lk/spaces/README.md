---
title: AeroVision LK
emoji: 🚁
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

# AeroVision LK — Aerial Vehicle Detection

YOLOv8s + SAHI aerial vehicle detection on VisDrone imagery. Compare Standard YOLO (single 640px pass) vs SAHI (512px tiled inference) for small-object detection.

- **Model:** YOLOv8s INT8 ONNX (11 MB)
- **Dataset:** VisDrone2019-DET (9 classes)
- **Best mAP@0.5:** 54.44% (SAHI 512/0.1) — +10.36pp over baseline
- **Quantization:** 3.9× size reduction via INT8 dynamic quantization
