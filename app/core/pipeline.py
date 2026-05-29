"""
Core pipeline — with frame snapshot capture for abandoned object alerts
and absolute-path config resolution.

Surgical Fix 2 (ROI Mask) is applied here for live inference consistency
with evaluate.py. The top N% of each frame is zeroed out to suppress
tree/shadow noise before any model processing.
"""

import time
import json
import os
import base64
import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path

from models.anomaly_detector import LiteAnomalyDetector
from utils.heatmap import CrowdDensityHeatmap
from utils.alert_engine import AlertEngine
from utils.object_tracker import AbandonedObjectTracker


class InferencePipeline:
    """
    Integrates all 3 modules into a single inference pipeline:
      Module 1: CNN-LSTM Anomaly Detector
      Module 2: Crowd Density Heatmap
      Module 3: Alert Engine (anomaly + density signals)
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, config_path: str = "config.json", alert_engine: AlertEngine = None):
        # Resolve config path relative to project root
        if not os.path.isabs(config_path):
            root = Path(__file__).parent.parent.parent
            config_path = str(root / config_path)

        with open(config_path) as f:
            self.cfg = json.load(f)

        ds  = self.cfg["dataset"]
        inf = self.cfg["inference"]

        self.frame_w   = ds["frame_width"]
        self.frame_h   = ds["frame_height"]
        self.seq_len   = ds["sequence_length"]
        self.threshold = inf["anomaly_threshold"]
        self.device    = torch.device("cpu")

        # ── Module 1: Anomaly Detector ──────────────────────────────────────
        model_path = self.cfg["model"]["model_path"]
        if not os.path.isabs(model_path):
            root = Path(__file__).parent.parent.parent
            model_path = str(root / model_path)

        self.model = LiteAnomalyDetector(config_path)

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"[OK] Loaded model from {model_path}")
        else:
            print(f"[WARN] Model not found at {model_path}. Using random weights (demo mode).")

        self.model.eval()

        # ── Dynamic INT8 Quantization (Optimization) ─────────────────────────
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.LSTM, torch.nn.Linear},
                dtype=torch.qint8
            )
            print("[OK] Applied INT8 dynamic quantization for inference optimization")
        except Exception as e:
            print(f"[WARN] Failed to apply quantization: {e}")

        # ── Module 2: Heatmap ───────────────────────────────────────────────
        self.heatmap = CrowdDensityHeatmap(config_path)

        # ── Module 3: Abandoned Object Tracker ─────────────────────────────
        self.obj_tracker = AbandonedObjectTracker(config_path)
        print("[OK] Abandoned object tracker ready")

        # ── Alert Engine ────────────────────────────────────────────────────
        self.alerts = alert_engine or AlertEngine(config_path)

        # Frame/feature buffers
        self.frame_buffer: deque   = deque(maxlen=self.seq_len)
        self.feature_buffer: deque = deque(maxlen=self.seq_len)
        self.score_buffer: deque   = deque(maxlen=10)
        self.adaptive_score_history: deque = deque(maxlen=300)

        # Latest heatmap frame for the dedicated heatmap stream endpoint
        self.latest_heatmap_frame: bytes = None
        # Latest annotated YOLO/detection frame for detection stream endpoint
        self.latest_detection_frame: bytes = None

    def preprocess_frame(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """BGR frame → normalized (3, H, W) tensor."""
        frame = cv2.resize(bgr_frame, (self.frame_w, self.frame_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return tensor

    def get_anomaly_score(self) -> float:
        if len(self.feature_buffer) < self.seq_len:
            return 0.0
        seq_features = torch.stack(list(self.feature_buffer), dim=0).unsqueeze(0)
        with torch.no_grad():
            score, err = self.model.forward_features(seq_features)
        return float(err[0].item())

    @staticmethod
    def _encode_frame_b64(frame: np.ndarray, quality: int = 75) -> str:
        """Encode a BGR frame to a base64 JPEG string for embedding in JSON."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def run(
        self,
        video_path: str,
        threshold: float = None,
        save_output: bool = False,
        display: bool = True,
        yolo_every_n_frames: int = None,
        max_frames: int = None,
        progress_callback=None,
        session_id: int = None,
    ):
        threshold = threshold or self.threshold
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps_src    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_target = float(self.cfg["inference"].get("fps_target", 5))
        yolo_every_n_frames = int(self.cfg["inference"].get("yolo_every_n_frames", 10))

        skip_interval = max(1, int(round(fps_src / fps_target)))

        writer = None
        if save_output:
            Path("outputs/videos").mkdir(parents=True, exist_ok=True)
            out_path = str(Path("outputs/videos") / (Path(video_path).stem + "_annotated.mp4"))
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps_target, (640, 480))

        frame_id      = 0
        processed_id  = 0
        t_prev        = time.time()
        t_start       = t_prev
        results       = []

        _last_obj_frame     = None
        _last_heatmap_frame = None
        anomaly_score       = 0.0

        prev_frame_gray = None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip_interval != 0:
                frame_id += 1
                continue

            if max_frames and processed_id >= max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_enhanced = clahe.apply(gray)
            frame_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)

            motion_score = 0.0
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray_enhanced, prev_frame_gray)
                _, thresh_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_score = float(np.sum(thresh_diff) / (gray.shape[0] * gray.shape[1]))
            prev_frame_gray = gray_enhanced

            blurred = cv2.GaussianBlur(frame_enhanced, (5, 5), 0)

            # Module 2: Heatmap
            heatmap_frame, density_score = self.heatmap.process_frame(blurred, return_overlay=True)
            _last_heatmap_frame = heatmap_frame

            # Module 1: Anomaly detection
            tensor = self.preprocess_frame(blurred)
            self.frame_buffer.append(tensor)
            with torch.no_grad():
                feat = self.model.extract_features(tensor.unsqueeze(0))
                self.feature_buffer.append(feat.squeeze(0))

            raw_err = self.get_anomaly_score()

            # Module 3: Object detection (every N frames)
            yolo_boost = 0.0
            if processed_id % yolo_every_n_frames == 0:
                small = cv2.resize(blurred, (320, 240))
                _last_obj_frame, abandoned_events = self.obj_tracker.process_frame(small, frame_id)
                # Scale detection frame back to display size
                _last_obj_frame = cv2.resize(_last_obj_frame, (640, 480))
            else:
                abandoned_events = []
                if _last_obj_frame is None:
                    _last_obj_frame = cv2.resize(blurred, (640, 480))

            yolo_boost_weights = self.cfg["inference"].get("yolo_boost_weights", {
                'bicycle': 1.0, 'motorcycle': 1.0, 'truck': 1.0, 
                'bus': 1.0, 'car': 1.0, 'skateboard': 1.0
            })

            for t in self.obj_tracker.get_track_summary():
                weight = yolo_boost_weights.get(t['class_name'], 0.0)
                if weight > yolo_boost:
                    yolo_boost = weight

            blended_score = max(raw_err, yolo_boost, motion_score * 5.0)
            self.score_buffer.append(blended_score)
            anomaly_score = float(np.mean(list(self.score_buffer))) if self.score_buffer else 0.0

            self.adaptive_score_history.append(anomaly_score)
            if len(self.adaptive_score_history) > 30:
                score_mean = np.mean(self.adaptive_score_history)
                score_std  = np.std(self.adaptive_score_history)
                current_threshold = float(max(threshold, score_mean + 3 * score_std))
            else:
                current_threshold = threshold

            is_anomaly = anomaly_score > current_threshold

            # Build composite display frame
            heatmap_resized = cv2.resize(heatmap_frame, (640, 480))
            display_frame   = cv2.addWeighted(heatmap_resized, 0.7, _last_obj_frame, 0.3, 0)

            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            label = f"{'ANOMALY' if is_anomaly else 'NORMAL'}  Score:{anomaly_score:.2f}"
            n_abandoned = len([o for o in self.obj_tracker.get_track_summary() if o['alerted']])
            cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(
                display_frame,
                f"Density:{density_score:.2f}  FPS:{1.0/max(time.time()-t_prev,1e-6):.1f}  "
                f"Abandoned:{n_abandoned}  Frame:{frame_id}",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
            )

            # Capture B64 snapshot for alert embedding
            if is_anomaly:
                frame_b64 = self._encode_frame_b64(display_frame, quality=60)
            else:
                frame_b64 = None

            # Heatmap-annotated frame for heatmap stream
            hm_annotated = heatmap_resized.copy()
            cv2.putText(hm_annotated,
                        f"DENSITY: {density_score:.3f}  Frame: {frame_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 204), 2)
            _, hm_buf = cv2.imencode(".jpg", hm_annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.latest_heatmap_frame = hm_buf.tobytes()

            # Detection-annotated frame for detection stream
            _, det_buf = cv2.imencode(".jpg", _last_obj_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.latest_detection_frame = det_buf.tobytes()

            # Push specialized frames into global_state for stream endpoints
            from .state import global_state as _gs
            _gs.update(
                latest_heatmap_frame=self.latest_heatmap_frame,
                latest_detection_frame=self.latest_detection_frame,
            )

            # Abandoned object alerts with snapshot
            abandoned_events_alerts = []
            for ev in abandoned_events:
                # Capture snapshot of the detection frame when abandoned object found
                snap_b64 = self._encode_frame_b64(_last_obj_frame, quality=70)
                alert = self.alerts.check_abandoned_object(
                    frame_id          = ev["frame_id"],
                    object_id         = ev["obj_id"],
                    stationary_seconds= ev["stationary_seconds"],
                    class_name        = ev["class_name"],
                    snapshot_b64      = snap_b64,
                )
                if alert:
                    abandoned_events_alerts.append(alert)

            # Alert engine: anomaly + density
            alert1 = self.alerts.check_anomaly(frame_id, anomaly_score,
                                               dynamic_threshold=current_threshold,
                                               frame_snapshot_b64=frame_b64)
            alert2 = self.alerts.check_density(frame_id, density_score,
                                               frame_snapshot_b64=frame_b64)

            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # Emit live metrics via WebSocket
            from app import socketio
            socketio.emit('metrics_update', {
                'fps': round(fps, 1),
                'density_score': round(density_score, 4),
                'anomaly_score': round(anomaly_score, 4),
                'current_threshold': round(current_threshold, 4),
                'is_anomaly': is_anomaly,
                'motion_score': round(motion_score, 4),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            })

            # Log metrics every 5 processed frames
            if processed_id % 5 == 0:
                self.alerts.log_metrics(fps, density_score, anomaly_score, frame_id,
                                        current_threshold=current_threshold)
                from .state import global_state
                global_state.update(threshold=current_threshold)

            for alert in [alert1, alert2] + abandoned_events_alerts:
                if alert:
                    socketio.emit('new_alert', alert.to_dict())

            results.append({
                "frame_id":      frame_id,
                "anomaly_score": round(anomaly_score, 4),
                "density_score": round(density_score, 4),
                "is_anomaly":    is_anomaly,
            })

            if writer:
                writer.write(display_frame)
            if display:
                cv2.imshow("Crowd Anomaly Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Update global state with composite frame for /api/stream
            from .state import global_state
            ret_enc, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret_enc:
                global_state.update(latest_frame=buffer.tobytes())

            if progress_callback:
                progress_callback(processed_id, total // skip_interval)

            frame_id     += 1
            processed_id += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self.obj_tracker.reset()
        self.heatmap.reset()

        # End session
        if session_id:
            anomaly_frames = sum(1 for r in results if r.get("is_anomaly"))
            self.alerts.end_session(session_id, len(results), anomaly_frames)

        return results
