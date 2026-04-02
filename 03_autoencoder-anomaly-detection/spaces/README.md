---
title: Autoencoder Anomaly Detection
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

# Autoencoder Anomaly Detection

Upload a hazelnut image to detect surface defects using a convolutional autoencoder trained on MVTec AD.

- **Architecture:** ConvAutoencoder (778K params)
- **Dataset:** MVTec AD hazelnut (train on normal only)
- **Best threshold:** p95 = 0.000470
- **PR-AUC:** 0.93 | **F1:** 0.65
