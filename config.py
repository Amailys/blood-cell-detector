from pathlib import Path

BASE_RESULTS_DIR = Path("/home/amailyss/scai/stage-lizards-iees/results/yolo_detect_classify_seed")
DATA_YAML_PATH = Path("/home/amailyss/scai/stage-lizards-iees/codes/yolo_detect_classify/data_yolo.yaml")
VAL_IMAGES_PATH = Path("/home/amailyss/scai/stage-lizards-iees/datasets/split_data_for_yolo/images/val")
VAL_LABELS_PATH = Path("/home/amailyss/scai/stage-lizards-iees/datasets/split_data_for_yolo/labels/val")

TEST_IMAGES_PATH = Path("/home/amailyss/scai/stage-lizards-iees/datasets/split_data_for_yolo/images/test")
TEST_LABELS_PATH = Path("/home/amailyss/scai/stage-lizards-iees/datasets/split_data_for_yolo/labels/test")


EXP_NAME = "train"
SEEDS = [7,42,99,123,2025] 
MODEL_NAME = "yolov8s.pt" 

TRAIN_PARAMS = {
    "epochs": 300,
    "img_size": 960,
    "batch_size": 4,
    "workers": 4,
    "optimizer": 'auto',
    "lr0": 0.01,
    "freeze": None,
    "patience":100,
    
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0
}

VAL_PARAMS = {
    "img_size": 960,
    "batch_size": 4,
    "workers": 4,
    "iou_thresh": 0.5,
    "iou" : 0.7,
    "conf" : 0.25
}

