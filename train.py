from pathlib import Path
from ultralytics import YOLO


def train_model(
    model_name: str,
    data_yaml_path: Path,
    run_dir: Path,
    seed: int,
    train_params: dict
):
    """
    Train a YOLO model for a given configuration.

    Args:
        model_name (str): Base model name (e.g. "yolov8s.pt").
        data_yaml_path (Path): Path to data.yaml.
        run_dir (Path): Folder where this run's results will be saved.
        seed (int): Random seed for reproducibility.
        train_params (dict): Dict of all training hyperparameters.

    Returns:
        Path: Path to the best saved weights (best.pt).
    """
    print(f"🔹 Starting training for seed {seed}...")
    print(f"   - Params: {train_params}")

    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml_path),
        project=str(run_dir),
        name="train",
        seed=seed,

        epochs=train_params.get("epochs", 100),
        imgsz=train_params.get("img_size", 640),
        batch=train_params.get("batch_size", 4),
        workers=train_params.get("workers", 4),
        optimizer=train_params.get("optimizer", "auto"),
        lr0=train_params.get("lr0", 0.01),
        freeze=train_params.get("freeze", None),
        patience=train_params.get("patience", 100),

        hsv_h=train_params.get("hsv_h", 0.015),
        hsv_s=train_params.get("hsv_s", 0.7),
        hsv_v=train_params.get("hsv_v", 0.4),
        degrees=train_params.get("degrees", 0.0),
        translate=train_params.get("translate", 0.1),
        scale=train_params.get("scale", 0.5),
        shear=train_params.get("shear", 0.0),
        perspective=train_params.get("perspective", 0.0),
        flipud=train_params.get("flipud", 0.0),
        fliplr=train_params.get("fliplr", 0.5),
        mosaic=train_params.get("mosaic", 1.0),
        mixup=train_params.get("mixup", 0.0),
        copy_paste=train_params.get("copy_paste", 0.0),
    )

    best_model_path = Path(results.save_dir) / "weights/best.pt"
    print(f"✓ Training finished for seed {seed}.")
    print(f"   - Best model saved at: {best_model_path}")

    return best_model_path
