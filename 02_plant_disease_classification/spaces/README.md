---
title: Plant Disease Classification
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

# Plant Disease Classification

Upload a leaf image to classify plant diseases using a ResNet50 model fine-tuned on PlantVillage.

- **Architecture:** ResNet50 (25.6M params, two-stage fine-tuned)
- **Dataset:** PlantVillage (54,305 images, 38 classes, 14 crop species)
- **Test Accuracy:** 99.66% | **Macro-F1:** 0.9952
