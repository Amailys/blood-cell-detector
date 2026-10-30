import os
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import base64, cv2, numpy as np

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/best.pt"))

model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names


def predict(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image")

    results = model.predict(img, imgsz=960, conf=0.25, iou=0.7, verbose=False)
    result = results[0]

    boxes = result.boxes
    classes_detected = [int(c) for c in boxes.cls.tolist()]
    confidences = boxes.conf.tolist()

    counts = Counter(classes_detected)
    summary = {
        CLASS_NAMES[cls_id]: count
        for cls_id, count in sorted(counts.items())
    }

    mean_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0

    annotated = result.plot(line_width=1)
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "total_cells": len(classes_detected),
        "mean_confidence": mean_conf,
        "counts_per_class": summary,
        "annotated_image": img_base64
    }
