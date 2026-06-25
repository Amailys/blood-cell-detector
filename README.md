# 🔬 Automatic Cell Recognition System for Lizard Blood Smears

Automatic detection and classification of blood cells on lizard blood smear images using YOLOv8. The model detects up to ~300 cells per image across 9 classes, exposed via a REST API with an interactive web interface.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)
![Gradio](https://img.shields.io/badge/Gradio-6.19-orange)

---

## 🎯 Project Overview

This project was developed during an internship focused on hematology analysis for lizard species. The goal is to automate the identification of blood cell types on microscopy images of blood smears, a task traditionally done manually by biologists.

The full pipeline covers:
- Model training and evaluation (multi-seed, reproducible)
- REST API deployment with FastAPI
- Interactive web interface with Gradio

---

## 🧬 Detected Cell Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | RBC | Red Blood Cells (erythrocytes) |
| 1 | Eosinophil | Granulocyte with eosinophilic granules |
| 2 | Basophil | Granulocyte with basophilic granules |
| 3 | Monocyte | Large mononuclear leukocyte |
| 4 | Lymphocyte | Mononuclear leukocyte |
| 5 | Parasitized RBC | Red blood cell infected by a parasite |
| 6 | Azurophil | Granulocyte specific to reptiles |
| 7 | Thrombocyte | Platelet equivalent in reptiles |
| 8 | Heterophil | Reptile equivalent of neutrophils |

---

## 🏗️ Architecture

```
Blood smear image
        ↓
  FastAPI /predict
        ↓
  YOLOv8s (960px)
        ↓
┌───────────────────────────┐
│ • Annotated image (bboxes)│
│ • Cell count per class    │
│ • Mean confidence score   │
└───────────────────────────┘
        ↓
  Gradio web interface
```

---

## 📁 Project Structure

```
blood-cell-detector/
├── src/
│   ├── model.py        # Model loading and inference
│   ├── api.py          # FastAPI endpoints
├── app.py              # Gradio web interface
├── requirements.txt
└── README.md
```

Training and evaluation scripts:
```
├── config.py           # Hyperparameters and paths
├── train.py            # YOLOv8 training pipeline
├── val.py              # Evaluation (detection + classification metrics)
└── main.py             # Full experiment runner (multi-seed)
```

---

## ⚙️ Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv8s |
| Input size | 960 × 960 px |
| Classes | 9 |
| Training epochs | 100 |
| Confidence threshold | 0.25 |
| NMS IoU threshold | 0.7 |
| Training seeds | 5 (7, 42, 99, 123, 2025) |

The training pipeline runs over multiple seeds for robust evaluation. Metrics are reported with mean ± std across seeds, separating detection (localization) and classification performance.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Amailys/Academic-projects.git
cd "Academic-projects/Development of an Automatic Cell Recognition System for Lizard Blood Smears"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your model weights

Place your `best.pt` file in the project root (not included in this repository for confidentiality reasons):

```bash
# Update the MODEL_PATH in src/model.py to point to your weights
MODEL_PATH = "/path/to/your/best.pt"
```

### 4. Start the API

```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Start the web interface

In a second terminal:

```bash
python app.py
```

Open your browser at **http://127.0.0.1:7860**

---

## 🌐 API Reference

### `POST /predict`

Accepts a blood smear image and returns cell detections.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@blood_smear.jpg"
```

**Response:**
```json
{
  "total_cells": 296,
  "mean_confidence": 0.924,
  "counts_per_class": {
    "RBC": 287,
    "Lymphocyte": 2,
    "Parasitized RBC": 7
  },
  "annotated_image": "<base64 encoded image>"
}
```

### `GET /health`

Returns API status.

---

## 📊 Results

Evaluated on the test set over 5 random seeds (mean ± std).

| Class | Precision | Recall | F1-Score | mAP@.50 | mAP@.50-.95 |
|-------|-----------|--------|----------|---------|-------------|
| RBC | 0.99 ± 0.00 | 0.94 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 | 0.95 ± 0.00 |
| Eosinophil | 0.20 ± 0.27 | 0.13 ± 0.18 | 0.16 ± 0.22 | 0.20 ± 0.27 | 0.20 ± 0.27 |
| Basophil | 0.62 ± 0.27 | 0.50 ± 0.00 | 0.53 ± 0.11 | 0.62 ± 0.08 | 0.57 ± 0.06 |
| Monocyte | 0.75 ± 0.05 | 0.69 ± 0.05 | 0.72 ± 0.04 | 0.75 ± 0.03 | 0.68 ± 0.03 |
| Lymphocyte | 0.83 ± 0.03 | 0.78 ± 0.05 | 0.81 ± 0.02 | 0.83 ± 0.03 | 0.68 ± 0.02 |
| Parasitized RBC | 0.92 ± 0.03 | 0.76 ± 0.04 | 0.83 ± 0.03 | 0.86 ± 0.03 | 0.65 ± 0.02 |
| Azurophil | 0.15 ± 0.11 | 0.37 ± 0.22 | 0.21 ± 0.14 | 0.12 ± 0.08 | 0.11 ± 0.08 |
| Thrombocyte | 0.74 ± 0.06 | 0.51 ± 0.04 | 0.60 ± 0.04 | 0.66 ± 0.03 | 0.38 ± 0.01 |
| Heterophil | 0.97 ± 0.03 | 0.90 ± 0.04 | 0.93 ± 0.03 | 0.95 ± 0.02 | 0.89 ± 0.02 |
| **Macro Avg** | **0.69 ± 0.05** | **0.62 ± 0.03** | **0.64 ± 0.04** | **0.66 ± 0.04** | **0.57 ± 0.04** |

> Results averaged over 5 seeds for robustness. High variance on rare classes (Eosinophil, Azurophil) reflects their low representation in the dataset.

The evaluation pipeline (`val.py`) computes separate metrics for detection (localization) and classification, with a NaN-aware macro average to avoid bias from absent classes.

---

## 🔒 Note on Model Weights

The trained model weights (`best.pt`) are not included in this repository for confidentiality reasons related to the internship data. The code is fully functional with any compatible YOLOv8 detection model.

---

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://gradio.app/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
