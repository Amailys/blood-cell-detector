# 🔬 Automatic Cell Recognition System for Lizard Blood Smears

Automatic detection and classification of blood cells on lizard blood smear images using YOLOv8. The model detects up to ~300 cells per image across 9 classes, exposed via a REST API with an interactive web interface, and packaged with Docker.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)
![Gradio](https://img.shields.io/badge/Gradio-6.19-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## 🎯 Project Overview

This project was developed during an internship focused on hematology analysis for lizard species. The goal is to automate the identification of blood cell types on microscopy images of blood smears, a task traditionally done manually by biologists.

The full pipeline covers:
- Model training and evaluation (multi-seed, reproducible)
- REST API deployment with FastAPI
- Interactive web interface with Gradio
- Containerization with Docker Compose (one container for the API, one for the interface)

<img width="1203" height="715" alt="Capture d’écran 2026-06-25 à 14 05 13" src="https://github.com/user-attachments/assets/c68de73c-0e0f-4c89-8880-b950ffe91ea9" />

---

## 🧬 Detected Cell Classes

| ID | Class |
|----|-------|
| 0 | RBC |
| 1 | Eosinophil |
| 2 | Basophil |
| 3 | Monocyte |
| 4 | Lymphocyte |
| 5 | Parasitized RBC |
| 6 | Azurophil |
| 7 | Thrombocyte |
| 8 | Heterophil |

---

## 🏗️ Architecture

```
Blood smear image
        ↓
  Gradio web interface  (container "app", port 7860)
        ↓  HTTP POST
  FastAPI /predict      (container "api", port 8000)
        ↓
  YOLOv8s (960px, CPU inference)
        ↓
┌───────────────────────────┐
│ • Annotated image (bboxes)│
│ • Cell count per class    │
│ • Mean confidence score   │
└───────────────────────────┘
```

---

## 📁 Project Structure

```
/
├── config.py                   # Hyperparameters and paths
├── train.py                    # YOLOv8 training pipeline
├── val.py                      # Evaluation
├── run.py                      # Full experiment runner (multi-seed)
├── data_yolo.yaml              # Dataset configuration
├── blood-cell-detector/        # Deployment
│   ├── src/
│   │   ├── model.py            # Model loading and inference
│   │   └── api.py              # FastAPI endpoints
│   ├── app.py                  # Gradio web interface
│   ├── Dockerfile.api          # Image for the API (PyTorch CPU + YOLO)
│   ├── Dockerfile.app          # Image for the Gradio interface
│   ├── docker-compose.yml      # Runs both containers
│   ├── requirements-api.txt
│   └── requirements-app.txt
└── README.md
```

---

## ⚙️ Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv8s |
| Input size | 960 × 960 px |
| Classes | 9 |
| Training epochs | 300 |
| Confidence threshold | 0.25 |
| NMS IoU threshold | 0.7 |
| Training seeds | 7, 42, 99, 123, 2025 |

The training pipeline runs over multiple seeds for robust evaluation. Metrics are reported with mean ± std across seeds.

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

> Results averaged over 5 seeds for robustness. High variance on rare classes (Eosinophil, Azurophil, Basophil) reflects their low representation in the dataset.

---

## Model Weights

The trained weights (`best.pt`) are **not included** in this repository for confidentiality reasons related to the internship data. The code works with any compatible YOLOv8 detection model.

Place your weights at:

```
blood-cell-detector/models/yolo_detect_classify/best.pt
```

The path can be changed with the `MODEL_PATH` environment variable.

---

## 🐳 Run with Docker (recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Docker Compose)
- Your `best.pt` file

### 1. Clone the repository

```bash
git clone https://github.com/Amailys/blood-cell-detector.git
cd blood-cell-detector/blood-cell-detector
```

### 2. Add your model weights

```bash
mkdir -p models/yolo_detect_classify
cp /path/to/your/best.pt models/yolo_detect_classify/
```

The `models/` folder is mounted read-only into the API container and is excluded from Git and from the Docker image.

### 3. Build and start

```bash
docker compose up --build
```

### 4. Use the application

- Web interface: **http://localhost:7860**
- API: **http://localhost:8000** 


### 5. Stop

Press `Ctrl+C`, then:

```bash
docker compose down
```

### What Docker runs

| Service | Image | Port | Role |
|---------|-------|------|------|
| `api` | `Dockerfile.api` | 8000 | FastAPI + YOLOv8 inference |
| `app` | `Dockerfile.app` | 7860 | Gradio interface, calls the API |

The API image installs the CPU-only version of PyTorch (much lighter than the default CUDA build). The interface image only needs Gradio, `requests` and Pillow.

---

## Run without Docker

From the `blood-cell-detector/` folder:

### 1. Install dependencies

```bash
pip install torch torchvision
pip install -r requirements-api.txt -r requirements-app.txt
```

### 2. Start the API

```bash
MODEL_PATH=/path/to/your/best.pt uvicorn src.api:app --port 8000
```

### 3. Start the web interface

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

Returns `400` if the file is not a readable image.

### `GET /health`

Returns API status (`{"status": "ok"}`).

---

## 🏋️ Training and Evaluation

Edit the paths and hyperparameters in `config.py` and `data_yolo.yaml` to match your dataset, then run the multi-seed experiment:

```bash
python run.py
```

For each seed, the script trains a YOLOv8s model, validates it, and finally writes a mean ± std report across seeds.

---

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://gradio.app/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)
