import os
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from ultralytics.utils.ops import xywhn2xyxy
from torchvision.ops import box_iou
from tqdm import tqdm
from contextlib import redirect_stdout
from collections import Counter

def load_yolo_labels(label_path, img_width, img_height):
    """
    Load YOLO labels from a .txt file and convert them to xyxy box format.
    """
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return torch.empty((0, 4)), torch.empty(0, dtype=torch.int64)
    labels = np.loadtxt(label_path).reshape(-1, 5)
    cls = torch.tensor(labels[:, 0], dtype=torch.int64)
    boxes = torch.tensor(labels[:, 1:], dtype=torch.float32)
    boxes_xyxy = xywhn2xyxy(boxes, w=img_width, h=img_height)
    return boxes_xyxy, cls


def compute_ap_detection_coco(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_thresh=0.5, plot_curve=False, save_path=None):
    """
    Compute Average Precision (AP) using the COCO method for a given IoU threshold.
    """
    gt_data_per_img, pred_data, total_gt_count = {}, [], 0
    
    for img_id, (gt, pred, score) in enumerate(zip(all_gt_boxes, all_pred_boxes, all_pred_scores)):
        gt_numpy, pred_numpy, score_numpy = gt.numpy(), pred.numpy(), score.numpy()
        
        if len(gt_numpy) > 0:
            gt_data_per_img[img_id] = {'boxes': gt_numpy, 'detected': set()}
            total_gt_count += len(gt_numpy)
        
        for pred_box, pred_score in zip(pred_numpy, score_numpy):
            pred_data.append({'img_id': img_id, 'box': pred_box, 'score': pred_score})
            
    if not pred_data or total_gt_count == 0:
        return 0.0
        
    pred_data.sort(key=lambda x: x['score'], reverse=True)
    tp, fp = np.zeros(len(pred_data)), np.zeros(len(pred_data))
    
    for pred_idx, pred in enumerate(pred_data):
        gt_info = gt_data_per_img.get(pred['img_id'])
        if not gt_info or len(gt_info['boxes']) == 0:
            fp[pred_idx] = 1
            continue
            
        ious = box_iou(torch.tensor(pred['box']).unsqueeze(0), torch.tensor(gt_info['boxes']))[0].numpy()
        max_iou_idx = ious.argmax()

        if ious.max() >= iou_thresh and max_iou_idx not in gt_info['detected']:
            tp[pred_idx] = 1
            gt_info['detected'].add(max_iou_idx)
        else:
            fp[pred_idx] = 1
            
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / (total_gt_count + 1e-8)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    recall_101 = np.linspace(0., 1., 101)
    precision_101 = np.zeros_like(recall_101)

    for i, r_101 in enumerate(recall_101):
        mask = recall >= r_101
        if np.any(mask):
            precision_101[i] = np.max(precision[mask])
    
    ap = np.mean(precision_101)
    
    if plot_curve and save_path:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'o', markersize=3, label='Raw data points')
        plt.plot(recall_101, precision_101, '-', label=f'Interpolated curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (IoU > {iou_thresh})')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(save_path)
        plt.close()

    return ap

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
    Evaluate a YOLO model for a given seed, save a complete report,
    and return metrics for unbiased macro aggregation (NaN-aware).

    Args:
        model_path (Path): path to best.pt
        val_dir (Path): output folder for reports/figures
        data_yaml_path (Path): data.yaml
        val_images_path (Path): images split
        val_labels_path (Path): labels split
        seed (int): seed
        val_params (dict): {"img_size","batch_size","workers","iou_thresh","iou","conf","split"}

    Returns:
        tuple(dict, dict): (metrics_results, model.names)
    """
    REPORT_PATH = val_dir / "report_validation.txt"
    VAL_NAME = "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    img_size   = val_params.get("img_size", 960)
    batch_size = val_params.get("batch_size", 4)
    workers    = val_params.get("workers", 4)
    iou_thresh = val_params.get("iou_thresh", 0.5)   
    iou        = val_params.get("iou", 0.7)          
    conf       = val_params.get("conf", 0.001)
    split      = val_params.get("split", "val")     

    metrics_results = {}

    with open(REPORT_PATH, 'w') as f, redirect_stdout(f):
        print(f"🔹 Seed: {seed}, Image Size: {img_size}, IoU match thresh (manual): {iou_thresh}, "
              f"NMS IoU: {iou}, Conf: {conf}, Split: {split}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if not model_path.exists():
            print(f"❌ Error: model file does not exist at {model_path}")
            return None, None

        model = YOLO(str(model_path))
        
        # --- 1) YOLO.val report ---
        print("\n📊 YOLO global report ---")
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
        yolo_metrics_per_class['Mean'] = {'p': mean_p, 'r': mean_r, 'f1': mean_f1, 'ap50': mean_map50, 'ap': mean_map}
        print(f"{'Mean':<25} | {mean_p:<10.4f} | {mean_r:<10.4f} | {mean_f1:<10.4f} | {mean_map50:<10.4f} | {mean_map:<12.4f}")
        
        metrics_results['yolo_report'] = yolo_metrics_per_class

        # --- 2) Manual computation (detection vs classification) ---
        print("\n🔎 Manual computation...")
        val_results = list(model.predict(
            source=str(val_images_path), 
            imgsz=img_size, 
            batch=batch_size, 
            conf=conf,
            iou=iou, 
            verbose=False, 
            stream=True
        ))
        
        all_true_cls, all_pred_cls = [], []
        tp_det, fp_det, fn_det = 0, 0, 0
        all_gt_boxes, all_pred_boxes, all_pred_scores = [], [], []

        for r in tqdm(val_results, desc=f"Manual computations Seed {seed}", leave=False):
            try:
                img_height, img_width = r.orig_shape
                base_name = os.path.splitext(os.path.basename(r.path))[0]
                label_file = os.path.join(str(val_labels_path), f"{base_name}.txt")

                gt_boxes, gt_cls = load_yolo_labels(label_file, img_width, img_height)
                
                pred_boxes, pred_cls, pred_scores = r.boxes.xyxy, r.boxes.cls, r.boxes.conf

                all_gt_boxes.append(gt_boxes.cpu())
                all_pred_boxes.append(pred_boxes.cpu())
                all_pred_scores.append(pred_scores.cpu())

                device = pred_boxes.device if len(pred_boxes) > 0 else "cpu"
                num_gt = len(gt_boxes)
                num_pred = len(pred_boxes)

                if num_pred == 0:
                    fn_det += num_gt
                    continue
                if num_gt == 0:
                    fp_det += num_pred
                    continue

                sorted_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes_sorted = pred_boxes[sorted_idx]
                pred_cls_sorted   = pred_cls[sorted_idx]

                iou_matrix = box_iou(pred_boxes_sorted, gt_boxes.to(device))
                used_gt = torch.zeros(num_gt, dtype=torch.bool, device=device)
                tp_in_image, fp_in_image = 0, 0

                for p_idx in range(num_pred):
                    ious = iou_matrix[p_idx]
                    ious_masked = ious.clone()
                    ious_masked[used_gt] = -1
                    
                    best_iou, best_gt_idx = torch.max(ious_masked, dim=0)

                    if best_iou >= iou_thresh:
                        tp_in_image += 1
                        used_gt[best_gt_idx] = True
                        all_true_cls.append(int(gt_cls[best_gt_idx]))
                        all_pred_cls.append(int(pred_cls_sorted[p_idx]))
                    else:
                        fp_in_image += 1

                fn_in_image = (~used_gt).sum().item()

                tp_det += tp_in_image
                fp_det += fp_in_image
                fn_det += fn_in_image

            except Exception as e:
                print(f"\n⚠️ Error on image {r.path}: {e}. Image skipped.")

        # PR curve & mAP
        map_05_path = val_dir / "pr_curve_map_05.png"
        map_05 = compute_ap_detection_coco(
            all_gt_boxes, all_pred_boxes, all_pred_scores,
            iou_thresh=0.5, plot_curve=True, save_path=map_05_path
        )
        map_05_095 = np.mean([
            compute_ap_detection_coco(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_thresh=t)
            for t in np.arange(0.5, 1.0, 0.05)
        ])
        
        # --- 3) Detection report ---
        precision_det = tp_det / (tp_det + fp_det + 1e-8)
        recall_det    = tp_det / (tp_det + fn_det + 1e-8)
        f1_det        = 2 * (precision_det * recall_det) / (precision_det + recall_det + 1e-8)

        print("\n🎯 DETECTION metrics (localization)")
        header = f"{'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<7} {'Recall':<7} {'F1':<7} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12}"
        print(header)
        print("-" * (len(header) + 2))
        print(f"{tp_det:<5} {fp_det:<5} {fn_det:<5} {precision_det:<7.4f} {recall_det:<7.4f} {f1_det:<7.4f} {map_05:<10.4f} {map_05_095:<12.4f}")
        
        metrics_results['detection_report'] = {
            'TP': tp_det, 'FP': fp_det, 'FN': fn_det,
            'Prec': precision_det, 'Recall': recall_det, 'F1': f1_det,
            'mAP@0.5': map_05, 'mAP@0.5:0.95': map_05_095
        }

        if len(all_true_cls) > 0:
            gt_counts_on_tp_crops = Counter(all_true_cls)
            class_names = model.names

            print("\n\n#####################################################################")
            print("📊 TABLE OF NUMBER OF GT BOXES PER CLASS (Recall Denominator)")
            print("#####################################################################")
            gt_header = f"{'Class':<20} | {'GT Count (Recall Denom.)':<25}"
            print(gt_header)
            print("-"*(len(gt_header) + 3))

            sorted_class_ids = sorted(class_names.keys())
            has_printed = False
            for class_id in sorted_class_ids:
                cls_name = class_names.get(class_id, f"Class {class_id}")
                count = gt_counts_on_tp_crops.get(class_id, 0)
                if count > 0:
                    print(f"{cls_name:<20} | {count:<25}")
                    has_printed = True
            if not has_printed:
                print(f"{'No GT in TP crops.':<20} | {'0':<25}")
            print("-"*(len(gt_header) + 3))
            print("#####################################################################\n")

        # --- 4) Classification report ---
        classification_metrics = {}
        if len(all_true_cls) > 0:
            class_names = model.names
            labels_for_cm = sorted(list(set(all_true_cls) | set(all_pred_cls)))

            precision_cls = precision_score(all_true_cls, all_pred_cls, labels=labels_for_cm, average=None, zero_division=np.nan)
            recall_cls    = recall_score(all_true_cls, all_pred_cls, labels=labels_for_cm, average=None, zero_division=np.nan)
            f1_cls        = f1_score(all_true_cls, all_pred_cls, labels=labels_for_cm, average=None, zero_division=np.nan)

            accuracy      = accuracy_score(all_true_cls, all_pred_cls)

            precision_macro = np.nanmean(precision_cls)
            recall_macro    = np.nanmean(recall_cls)
            f1_macro        = np.nanmean(f1_cls)

            print("\n🏷️ CLASSIFICATION metrics (on well-localized boxes)")
            print(f"{'Class':<20} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
            print("-"*45)

            cls_metrics_per_class = {}
            for i, label_id in enumerate(labels_for_cm):
                cls_name = class_names.get(label_id, f"Class {label_id}")
                p_val, r_val, f1_val = precision_cls[i], recall_cls[i], f1_cls[i]
                cls_metrics_per_class[cls_name] = {'Prec': p_val, 'Recall': r_val, 'F1': f1_val}

                p_str  = f"{p_val:<7.4f}" if not np.isnan(p_val) else f"{'N/A':<7}"
                r_str  = f"{r_val:<7.4f}" if not np.isnan(r_val) else f"{'N/A':<7}"
                f1_str = f"{f1_val:<7.4f}" if not np.isnan(f1_val) else f"{'N/A':<7}"
                print(f"{cls_name:<20} | {p_str} | {r_str} | {f1_str}")

            print("-"*45)
            cls_metrics_per_class['Macro Avg'] = {'Prec': precision_macro, 'Recall': recall_macro, 'F1': f1_macro}
            print(f"{'Macro Avg':<20} | {precision_macro:<7.4f} | {recall_macro:<7.4f} | {f1_macro:<7.4f}")
            print("-"*45)
            print(f"🎯 Global accuracy: {accuracy:.4f}")

            classification_metrics['per_class'] = cls_metrics_per_class
            classification_metrics['accuracy']  = accuracy

            cm = confusion_matrix(all_true_cls, all_pred_cls, labels=labels_for_cm)
            cm_float = cm.astype('float')
            row_sums = cm_float.sum(axis=1, keepdims=True)
            cm_normalized = np.where(row_sums > 0, cm_float / row_sums, np.nan)

            def plot_cm_yolo(cm_norm, labels, path, title):
                cm_display = np.nan_to_num(cm_norm, nan=0.0)
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm_display, annot=True, fmt='.1%', cmap='Blues', ax=ax)
                ax.set_title(title, fontsize=16)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels, rotation=0)
                plt.tight_layout()
                plt.savefig(path, dpi=300)
                plt.close(fig)

            display_labels = [class_names.get(label_id, f"ID {label_id}") for label_id in labels_for_cm]
            cm_path = val_dir / "confusion_matrix.png"
            plot_cm_yolo(cm_normalized, display_labels, cm_path,
                         f'Normalized Confusion Matrix (Classification, conf={conf})')
            print(f"\n✅ Confusion matrix saved at: {cm_path}")
        
        metrics_results['classification_report'] = classification_metrics
        
        if len(all_true_cls) > 0:
            metrics_results['raw_classification_data'] = {
                'cm_normalized': cm_normalized,
                'labels_for_cm': labels_for_cm
            }

    print(f"\n✅ Individual report for seed {seed} saved at: {REPORT_PATH}")
    return metrics_results, model.names


def generate_mean_report(all_results: list, class_names: dict, output_dir: Path, conf_for_title: float = None):
    """
    Generate an averaged report AND a MACRO-aggregated confusion matrix
    ignoring NaNs (absent classes), sorted by decreasing recall (diagonal).

    Args:
        all_results (list): list of metrics dicts for each run.
        class_names (dict): id -> class name.
        output_dir (Path): main experiment folder.
        conf_for_title (float|None): displayed in figure titles if provided.
    """
    output_path = output_dir / "mean_report.txt"

    with open(output_path, 'w') as f, redirect_stdout(f):
        print("="*80)
        if conf_for_title is not None:
            print(f"===== MEAN REPORT OVER ALL SEEDS - CONF={conf_for_title} =====")
        else:
            print("===== MEAN REPORT OVER ALL SEEDS =====")
        print("="*80)

        # --- 1) Averaged YOLO report ---
        yolo_data = []
        for run in all_results:
            for cls, metrics in run.get('yolo_report', {}).items():
                yolo_data.append({'Class': cls, **metrics})
        
        if yolo_data:
            df_yolo = pd.DataFrame(yolo_data)
            yolo_mean = df_yolo.groupby('Class').mean(numeric_only=True)
            yolo_std  = df_yolo.groupby('Class').std(numeric_only=True).fillna(0)

            print("\n📊 AVERAGED YOLO GLOBAL REPORT (mean ± std) ---")
            header = f"{'Class':<25} | {'Precision':<20} | {'Recall':<20} | {'F1-Score':<20} | {'mAP@.50':<20} | {'mAP@.50-.95':<20}"
            print(header)
            print("-" * (len(header) + 2))
            
            ordered_classes = list(class_names.values()) + ['Mean']
            for cls_name in ordered_classes:
                if cls_name in yolo_mean.index:
                    m, s = yolo_mean.loc[cls_name], yolo_std.loc[cls_name]
                    p_str    = f"{m['p']:.4f} ± {s['p']:.2f}"
                    r_str    = f"{m['r']:.4f} ± {s['r']:.2f}"
                    f1_str   = f"{m['f1']:.4f} ± {s['f1']:.2f}"
                    ap50_str = f"{m['ap50']:.4f} ± {s['ap50']:.2f}"
                    ap_str   = f"{m['ap']:.4f} ± {s['ap']:.2f}"
                    print(f"{cls_name:<25} | {p_str:<20} | {r_str:<20} | {f1_str:<20} | {ap50_str:<20} | {ap_str:<20}")

        # --- 2) Averaged detection ---
        det_data = [run['detection_report'] for run in all_results if 'detection_report' in run]
        if det_data:
            df_det = pd.DataFrame(det_data)
            det_mean = df_det.mean(numeric_only=True)
            det_std  = df_det.std(numeric_only=True).fillna(0)

            print("\n🎯 AVERAGED DETECTION METRICS (mean ± std)")
            header = f"{'TP':<12} {'FP':<12} {'FN':<12} {'Prec':<18} {'Recall':<18} {'F1':<18} {'mAP@0.5':<18} {'mAP@0.5:0.95':<18}"
            print(header)
            print("-" * (len(header) + 2))

            tp_str   = f"{det_mean['TP']:.1f} ± {det_std['TP']:.2f}"
            fp_str   = f"{det_mean['FP']:.1f} ± {det_std['FP']:.2f}"
            fn_str   = f"{det_mean['FN']:.1f} ± {det_std['FN']:.2f}"
            p_str    = f"{det_mean['Prec']:.4f} ± {det_std['Prec']:.2f}"
            r_str    = f"{det_mean['Recall']:.4f} ± {det_std['Recall']:.2f}"
            f1_str   = f"{det_mean['F1']:.4f} ± {det_std['F1']:.2f}"
            map50_str= f"{det_mean['mAP@0.5']:.4f} ± {det_std['mAP@0.5']:.2f}"
            map95_str= f"{det_mean['mAP@0.5:0.95']:.4f} ± {det_std['mAP@0.5:0.95']:.2f}"

            print(f"{tp_str:<12} {fp_str:<12} {fn_str:<12} {p_str:<18} {r_str:<18} {f1_str:<18} {map50_str:<18} {map95_str:<18}")

        # --- 3) Averaged classification ---
        classif_data = []
        accuracies = []
        for run in all_results:
            if 'classification_report' in run and run['classification_report']:
                accuracies.append(run['classification_report']['accuracy'])
                for cls, metrics in run['classification_report']['per_class'].items():
                    classif_data.append({'Class': cls, **metrics})

        if classif_data:
            df_cls = pd.DataFrame(classif_data)
            cls_mean = df_cls.groupby('Class').agg({
                'Prec': np.nanmean, 'Recall': np.nanmean, 'F1': np.nanmean
            })
            cls_std = df_cls.groupby('Class').agg({
                'Prec': np.nanstd, 'Recall': np.nanstd, 'F1': np.nanstd
            }).fillna(0)

            print("\n🏷️ AVERAGED CLASSIFICATION METRICS (mean ± std, unbiased)")
            header = f"{'Class':<20} | {'Prec':<18} | {'Recall':<18} | {'F1':<18}"
            print(header)
            print("-" * (len(header) + 2))

            if 'Macro Avg' in cls_mean.index:
                m_macro = cls_mean.loc['Macro Avg']
                s_macro = cls_std.loc['Macro Avg'].fillna(0)
                cls_mean = cls_mean.drop('Macro Avg')
                cls_std  = cls_std.drop('Macro Avg')
            else:
                m_macro = s_macro = None

            for cls_name in list(class_names.values()):
                if cls_name in cls_mean.index:
                    m, s = cls_mean.loc[cls_name], cls_std.loc[cls_name]
                    p_str  = f"{m['Prec']:.4f} ± {s['Prec']:.2f}"
                    r_str  = f"{m['Recall']:.4f} ± {s['Recall']:.2f}"
                    f1_str = f"{m['F1']:.4f} ± {s['F1']:.2f}"
                    print(f"{cls_name:<20} | {p_str:<18} | {r_str:<18} | {f1_str:<18}")

            print("-" * (len(header) + 2))
            if m_macro is not None:
                p_str  = f"{m_macro['Prec']:.4f} ± {s_macro['Prec']:.2f}"
                r_str  = f"{m_macro['Recall']:.4f} ± {s_macro['Recall']:.2f}"
                f1_str = f"{m_macro['F1']:.4f} ± {s_macro['F1']:.2f}"
                print(f"{'Macro Avg':<20} | {p_str:<18} | {r_str:<18} | {f1_str:<18}")

            print("-" * (len(header) + 2))
            if len(accuracies) > 0:
                acc_mean, acc_std = float(np.mean(accuracies)), float(np.std(accuracies))
                print(f"🎯 Global accuracy: {acc_mean:.4f} ± {acc_std:.2f}")

        # --- 4) MACRO-aggregated Confusion Matrix ---
        all_cms = [run['raw_classification_data']['cm_normalized'] for run in all_results
                   if 'raw_classification_data' in run and 'cm_normalized' in run['raw_classification_data']]

        if all_cms:
            if 'raw_classification_data' in all_results[0]:
                all_class_indices = all_results[0]['raw_classification_data'].get('labels_for_cm',
                                                                                  sorted(class_names.keys()))
            else:
                all_class_indices = sorted(class_names.keys())

            display_labels = [class_names[i] for i in all_class_indices]

            cm_macro_aggregated = np.nanmean(np.stack(all_cms), axis=0)

            tp_scores = np.diag(cm_macro_aggregated)
            tp_scores_for_sort = np.nan_to_num(tp_scores, nan=-1)
            sorted_indices = np.argsort(tp_scores_for_sort)[::-1]

            cm_sorted = cm_macro_aggregated[sorted_indices, :][:, sorted_indices]
            display_labels_sorted = [display_labels[i] for i in sorted_indices]

            def plot_cm_yolo(cm_norm, labels, path, title):
                cm_display = np.nan_to_num(cm_norm, nan=0.0)
                fig, ax = plt.subplots(figsize=(14, 12))
                sns.heatmap(cm_display,
                            annot=True,
                            fmt='.1%',
                            cmap='Blues',
                            ax=ax,
                            xticklabels=labels,
                            yticklabels=labels)
                ax.set_title(f'{title}' + (f' (conf={conf_for_title})' if conf_for_title is not None else ''), fontsize=16)
                ax.set_xlabel('Predicted Label ', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                plt.setp(ax.get_yticklabels(), rotation=0)
                plt.tight_layout()
                plt.savefig(path, dpi=300)
                plt.close(fig)

            cm_path = output_dir / "aggregated_confusion_matrix_MACRO.png"
            plot_cm_yolo(cm_sorted, display_labels_sorted, cm_path,
                         'MACRO-aggregated & Normalized Confusion Matrix (Sorted by Desc. Recall)')
            print(f"\n✅ MACRO-aggregated confusion matrix saved at: {cm_path}")

            with open(output_path, 'a') as f:
                f.write("\n\n" + "="*80 + "\n")
                f.write("===== MACRO-AGGREGATED CONFUSION MATRIX (Unbiased Recall Mean) =====\n")
                f.write("="*80 + "\n")
                f.write(f"\nConfusion matrix saved here:\n{cm_path}\n")

    print(f"\n✅ Mean report saved at: {output_path}")
