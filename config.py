from pathlib import Path

BASE_RESULTS_DIR = Path("/home/amailyss/scai/extra/results")
DATA_YAML_PATH = Path("/home/amailyss/scai/extra/codes/data_yolo.yaml")
VAL_IMAGES_PATH = Path("/home/amailyss/scai/extra/datasets/split_data/images/val")
VAL_LABELS_PATH = Path("/home/amailyss/scai/extra/datasets/split_data/labels/val")

TEST_IMAGES_PATH = Path("/home/amailyss/scai/extra/datasets/split_data/images/test")
TEST_LABELS_PATH = Path("/home/amailyss/scai/extra/datasets/split_data/labels/test")


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

