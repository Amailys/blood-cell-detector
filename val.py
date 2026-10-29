import pandas as pd
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from contextlib import redirect_stdout


def validate_model(
    model_path: Path,
    val_dir: Path,
    data_yaml_path: Path,
    val_images_path: Path,
    val_labels_path: Path,
    seed: int,
    val_params: dict
):
    """
    Evaluate a YOLO model for a given seed using native Ultralytics val()
    and return per-class metrics for mean/std aggregation.
    """
    REPORT_PATH = val_dir / "report_validation.txt"
    VAL_NAME = "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    img_size   = val_params.get("img_size", 960)
    batch_size = val_params.get("batch_size", 4)
    workers    = val_params.get("workers", 4)
    iou        = val_params.get("iou", 0.7)
    conf       = val_params.get("conf", 0.001)
    split      = val_params.get("split", "val")

    metrics_results = {}

    with open(REPORT_PATH, 'w') as f, redirect_stdout(f):
        print(f"Seed: {seed}, Image Size: {img_size}, NMS IoU: {iou}, Conf: {conf}, Split: {split}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if not model_path.exists():
            print(f"Error: Model file does not exist at {model_path}")
            return None, None

        model = YOLO(str(model_path))
        
        global_metrics = model.val(
            data=str(data_yaml_path),
            project=str(val_dir),
            name=VAL_NAME,
            batch=batch_size,
            workers=workers,
            imgsz=img_size,
            conf=conf,
            iou=iou,
            verbose=True,
            split=split
        )
                                    
        names = global_metrics.names
        metrics_box = global_metrics.box
        yolo_metrics_per_class = {}

        header = f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mAP@.50':<10} | {'mAP@.50-.95':<12}"
        print(header)
        print("-" * (len(header) + 2))

        for i, cls_name in names.items():
            if i < len(metrics_box.p):
                p, r, f1 = metrics_box.p[i], metrics_box.r[i], metrics_box.f1[i]
                ap50, ap = metrics_box.ap50[i], metrics_box.ap[i]
                yolo_metrics_per_class[cls_name] = {'p': p, 'r': r, 'f1': f1, 'ap50': ap50, 'ap': ap}
                print(f"{cls_name:<25} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {ap50:<10.4f} | {ap:<12.4f}")

        print("-" * (len(header) + 2))
        mean_p, mean_r, mean_f1 = metrics_box.p.mean(), metrics_box.r.mean(), metrics_box.f1.mean()
        mean_map50, mean_map = metrics_box.map50, metrics_box.map
        yolo_metrics_per_class['Macro Avg'] = {'p': mean_p, 'r': mean_r, 'f1': mean_f1, 'ap50': mean_map50, 'ap': mean_map}
        print(f"{'Macro Avg':<25} | {mean_p:<10.4f} | {mean_r:<10.4f} | {mean_f1:<10.4f} | {mean_map50:<10.4f} | {mean_map:<12.4f}")
        
        metrics_results['yolo_report'] = yolo_metrics_per_class

    return metrics_results, model.names


def generate_mean_report(all_results: list, class_names: dict, output_dir: Path, conf_for_title: float = None):
    """
    Generate a mean report across seeds with standard deviations based on native YOLO results.
    """
    output_path = output_dir / "mean_report.txt"

    with open(output_path, 'w') as f, redirect_stdout(f):
        print("="*80)
        print("===== MEAN REPORT ACROSS SEEDS =====")
        print("="*80)

        yolo_data = []
        for run in all_results:
            for cls, metrics in run.get('yolo_report', {}).items():
                yolo_data.append({'Classe': cls, **metrics})
        
        if yolo_data:
            df_yolo = pd.DataFrame(yolo_data)
            yolo_mean = df_yolo.groupby('Classe').mean(numeric_only=True)
            yolo_std  = df_yolo.groupby('Classe').std(numeric_only=True).fillna(0)

            print("\nGlobal YOLO Mean Report (mean ± std):")
            header = f"{'Class':<25} | {'Precision':<20} | {'Recall':<20} | {'F1-Score':<20} | {'mAP@.50':<20} | {'mAP@.50-.95':<20}"
            print(header)
            print("-" * (len(header) + 2))
            
            ordered_classes = list(class_names.values()) + ['Macro Avg']
            for cls_name in ordered_classes:
                if cls_name in yolo_mean.index:
                    m, s = yolo_mean.loc[cls_name], yolo_std.loc[cls_name]
                    p_str    = f"{m['p']:.4f} ± {s['p']:.2f}"
                    r_str    = f"{m['r']:.4f} ± {s['r']:.2f}"
                    f1_str   = f"{m['f1']:.4f} ± {s['f1']:.2f}"
                    ap50_str = f"{m['ap50']:.4f} ± {s['ap50']:.2f}"
                    ap_str   = f"{m['ap']:.4f} ± {s['ap']:.2f}"
                    print(f"{cls_name:<25} | {p_str:<20} | {r_str:<20} | {f1_str:<20} | {ap50_str:<20} | {ap_str:<20}")
