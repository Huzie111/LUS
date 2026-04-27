---
title: Lung Ultrasound AI
emoji: 🫁
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
---

# Lung Ultrasound AI Assistant

AI-powered lung ultrasound analysis for resource-constrained settings.

## Features
- Disease classification (COVID-19, Other Disease, Healthy)
- B-line detection with segmentation overlay
- Probability heatmap for explainability

## Model Details
- **Architecture**: EfficientNet-B3 + SegFormer
- **Training Data**: 1,463 images from Mulago and Kiruddu Hospitals, Uganda
- **Model Size**: 44 MB
- **Inference Time**: <200 ms on CPU

## Usage
1. Upload a lung ultrasound image
2. Wait for analysis
3. View classification and B-line detection results

## Citation
```bibtex
@article{serunjogi2026lung,
  title={Edge-Optimized Multi-Task Deep Learning for Lung Ultrasound Analysis},
  author={Serunjogi Huzaifa },
  journal={IEEE Access},
  year={2026}
}