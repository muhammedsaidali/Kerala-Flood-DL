---
title: Kerala Flood AI
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---
# 🌊 Kerala Flood Damage Assessment System

> **Multi-modal Deep Learning system for real-time flood damage classification using aerial imagery and weather data.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Overview

A production-grade end-to-end ML system that classifies flood damage severity from aerial/satellite imagery combined with real-time weather data. Built as a portfolio project demonstrating real ML engineering skills — from data pipeline to ONNX deployment.

**Damage Classes:**
| Class | Label | Description |
|-------|-------|-------------|
| 0 | ✅ No Damage | No visible flood impact |
| 1 | ⚠️ Minor Damage | Partial flooding, minimal structural impact |
| 2 | 🔶 Major Damage | Significant flooding, structural damage |
| 3 | 🔴 Catastrophic | Severe destruction, emergency response needed |

---

## 🧠 Model Architecture — FloodNet v2

```
Input: RGB Aerial Image (3×256×256) + Weather Vector (10-dim)
         │                                    │
   EfficientNet-B3                    Weather MLP
   (ImageNet pretrained)           [Linear → BN → ReLU]
         │                                    │
    Image Features (512)         Weather Features (128)
         └──────────┬────────────────┘
                    │  Cross-modal Fusion
                    │  [Concat → Linear → BN → ReLU → Dropout]
                    │
              Classifier Head
              [512 → 4 classes]
                    │
             Softmax Output
```

**Key specs:**
- **Backbone:** EfficientNet-B3 (ImageNet pretrained, fine-tuned)
- **Input:** RGB 256×256 aerial images + 10-dimensional weather features
- **Parameters:** 13.2M trainable
- **Training:** 10 epochs, FP16, OneCycleLR, WeightedRandomSampler
- **Best Val F1:** **0.9646** (macro, 4-class)
- **Inference:** ~12ms on CPU (ONNX Runtime)
- **Export:** ONNX opset 18, ~54MB

---

## 🌦️ Weather Features

Derived from **IMD (India Meteorological Department)** independent rainfall ranges per damage class:

| Feature | Description |
|---------|-------------|
| `rainfall_mm` | Current rainfall intensity (mm/hr) |
| `river_level_m` | River water level (metres) |
| `wind_speed_kmh` | Wind speed (km/h) |
| `humidity_pct` | Relative humidity (%) |
| `temp_c` | Temperature (°C) |
| `pressure_hpa` | Atmospheric pressure (hPa) |
| `duration_hrs` | Rainfall duration (hours) |
| `upstream_rainfall_mm` | Upstream catchment rainfall |
| `soil_moisture` | Soil saturation index (0–1) |
| `previous_flood_days` | Days since last flood event |

---

## 📁 Project Structure

```
kerala-flood-dl/
├── app.py                    # Streamlit dashboard (HF Spaces ready)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit dark theme config
├── model/
│   ├── floodnet.py           # FloodNet v2 architecture
│   ├── preprocess.py         # Data pipeline & augmentation
│   ├── train_kaggle.py       # Training script (Kaggle T4×2)
│   ├── floodnet.onnx         # Exported ONNX model (1.5 MB)
│   └── floodnet.onnx.data    # ONNX weights (52 MB)
├── backend/
│   ├── main.py               # FastAPI inference server
│   └── requirements.txt      # Backend dependencies
└── datasets/
    └── download_data.py      # Dataset download utilities
```

---

## 🚀 Quick Start

### Local Setup
```bash
git clone https://github.com/muhammedsaidali/Kerala-Flood-DL.git
cd Kerala-Flood-DL
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```
Open **http://localhost:8501**

### Kaggle Training
1. Upload `model/floodnet.py`, `model/preprocess.py`, `model/train_kaggle.py` to a Kaggle dataset
2. Add the [FloodNet UAV dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)
3. Enable T4×2 GPU accelerator
4. Run `train_kaggle.py` — completes in ~17 minutes

---

## 📊 Training Results

| Epoch | Train F1 | Val F1 | Val Acc |
|-------|----------|--------|---------|
| 1     | 0.2880   | 0.2870 | 36.0%   |
| 3     | 0.8099   | 0.5605 | 64.7%   |
| 5     | 0.9577   | 0.9646 | 98.5%   |
| 10    | 0.9887   | 0.9646 | 98.5%   |

**Per-class Val F1 (best epoch):**
- No Damage: 1.000
- Minor Damage: 1.000
- Major Damage: 0.941
- Catastrophic: 1.000

---

## 🗺️ Features

- **Live Weather** — Auto-fetches real-time data from Open-Meteo API for any Kerala district
- **Interactive Map** — Folium map with ISRO Bhuvan satellite tiles + CartoDB dark layer
- **14 Kerala Districts** — Pre-loaded coordinates for all districts
- **Historical Risk Chart** — Plotly bar chart of historical flood risk by district
- **ONNX Inference** — 12ms CPU inference, no GPU required at runtime
- **Rule-based Fallback** — Works even without ONNX model (demo mode)

---

## 🔧 Known Limitations

- **Class imbalance:** Only 2 "No Damage" training samples in the FloodNet UAV dataset — model rarely predicts class 0. Future work: collect more No Damage aerial images.
- **Text encoder:** DistilBERT trained but excluded from ONNX export for deployment efficiency (~540MB saved). Future work: quantize BERT for mobile deployment.
- **Dataset size:** 290 images total — production system would require thousands of labeled samples.

---

## 🛰️ Data Sources

| Source | Usage |
|--------|-------|
| [FloodNet UAV Dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) | Training images & masks |
| [Open-Meteo API](https://open-meteo.com) | Real-time weather data |
| [ISRO Bhuvan](https://bhuvan.nrsc.gov.in) | Satellite map tiles |
| IMD Rainfall Records | Weather feature normalization ranges |

---

## 🧪 Tech Stack

| Component | Technology |
|-----------|------------|
| Model Training | PyTorch 2.0, EfficientNet-B3, DistilBERT |
| Experiment Platform | Kaggle (Tesla T4×2, FP16) |
| Model Export | ONNX Runtime (opset 18) |
| Dashboard | Streamlit + Plotly + Folium |
| API Server | FastAPI + Uvicorn |
| Deployment | Hugging Face Spaces |

---

## 📄 License

MIT License — free to use for research and educational purposes.

---

<div align="center">
  <sub>Built with ❤️ from Kollam, Kerala · 8.8932°N, 76.6141°E</sub>
</div>
