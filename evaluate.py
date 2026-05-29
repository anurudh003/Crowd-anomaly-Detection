import os
import cv2
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from models.anomaly_detector import LiteAnomalyDetector
from app.core.pipeline import InferencePipeline

def parse_ground_truth(m_file_path):
    """Returns a list of binary arrays (length 200) for each test video."""
    gts = []
    with open(m_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'gt_frame' in line:
            gt_array = np.zeros(200, dtype=int)
            ranges_str = line.split('=')[1].strip().strip('[];')
            for r in ranges_str.split(','):
                parts = r.strip().split(':')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    gt_array[start - 1:end] = 1
                elif len(parts) == 1 and parts[0]:
                    gt_array[int(parts[0]) - 1] = 1
            gts.append(gt_array)
    return gts

def evaluate_model():
    base_dir = "data/raw/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
    m_file   = os.path.join(base_dir, "UCSDped1.m")

    print("Parsing ground truth...")
    gts = parse_ground_truth(m_file)
    print(f"Loaded ground truth for {len(gts)} videos.")

    print("Loading pipeline...")
    pipeline = InferencePipeline()
    
    # Reload un-quantized model for evaluation accuracy
    print("Disabling INT8 Quantization for benchmark accuracy...")
    model_path = pipeline.cfg["model"]["model_path"]
    pipeline.model = LiteAnomalyDetector("config.json")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        pipeline.model.load_state_dict(state)
        print(f"[OK] Loaded model from {model_path}")
    pipeline.model.eval()

    # Baseline smoothing
    SIGMA = 2.0

    all_scores = []
    all_labels = []

    num_videos = len(gts)

    for i in tqdm(range(num_videos), desc="Evaluating videos"):
        vid_dir = os.path.join(base_dir, f"Test{i + 1:03d}")
        if not os.path.exists(vid_dir):
            continue

        gt = gts[i]
        frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.tif')])
        pipeline.frame_buffer.clear()
        pipeline.obj_tracker.reset()

        vid_scores = []
        prev_frame_gray = None

        for frame_idx, f_name in enumerate(frame_files):
            f_path = os.path.join(vid_dir, f_name)
            frame = cv2.imread(f_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion energy
            motion_score = 0.0
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray, prev_frame_gray)
                _, thresh_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_score = np.sum(thresh_diff) / (gray.shape[0] * gray.shape[1])
            prev_frame_gray = gray

            # Model inference
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            tensor = pipeline.preprocess_frame(blurred)
            pipeline.frame_buffer.append(tensor)

            if len(pipeline.frame_buffer) == pipeline.seq_len:
                seq = torch.stack(list(pipeline.frame_buffer), dim=0).unsqueeze(0)
                with torch.no_grad():
                    head_score, recon_err = pipeline.model(seq)
                h_score = float(head_score[0].item())
            else:
                h_score = 0.0

            # YOLO Boost
            if frame_idx % 10 == 0:
                small = cv2.resize(frame, (320, 240))
                pipeline.obj_tracker.process_frame(small, frame_idx)

            yolo_boost = 0.0
            for t in pipeline.obj_tracker.get_track_summary():
                if t['class_name'] in ['bicycle', 'motorcycle', 'truck', 'bus', 'car', 'skateboard']:
                    yolo_boost = 1.0
                    break

            # Baseline Fusion: max logic
            final_score = max(h_score, yolo_boost, motion_score * 5.0)
            vid_scores.append(final_score)

        # Temporal smoothing
        vid_scores = gaussian_filter1d(vid_scores, sigma=SIGMA)
        
        all_scores.extend(vid_scores.tolist())
        all_labels.extend(gt.tolist())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Global min-max normalization
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-10)

    # Metrics
    auc = roc_auc_score(all_labels, all_scores)
    
    # Use the baseline threshold of 0.31
    threshold = 0.31
    all_preds = (all_scores > threshold).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n--- Computing Final Metrics ---")
    print(f"Optimal Threshold: {threshold}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Plots
    os.makedirs("outputs/plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend(); plt.savefig("outputs/plots/roc_curve.png"); plt.close()

if __name__ == "__main__":
    evaluate_model()
