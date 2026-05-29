import argparse
import time
import json
import os
import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path

from app.core.pipeline import InferencePipeline

def parse_args():
    p = argparse.ArgumentParser(description="Crowd Anomaly Detection Inference")
    p.add_argument("--video",     required=True,         help="Path to input video file")
    p.add_argument("--config",    default="config.json", help="Path to config JSON")
    p.add_argument("--threshold", type=float, default=None, help="Anomaly score threshold (overrides config)")
    p.add_argument("--save",      action="store_true",   help="Save annotated output video")
    p.add_argument("--no-display",action="store_true",   help="Disable live window (headless mode)")
    return p.parse_args()

    def run(
        self,
        video_path: str,
        threshold: float = None,
        save_output: bool = False,
        display: bool = True,
        yolo_every_n_frames: int = None,   # None = read from config
        max_frames: int = None,
        progress_callback=None,
    ):
        threshold = threshold or self.threshold
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps_src    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_target = float(self.cfg["inference"].get("fps_target", 5))
        # Read yolo cadence from config (default: run YOLO every 10 processed frames)
        yolo_every_n_frames = int(self.cfg["inference"].get("yolo_every_n_frames", 10))

        # How many source frames to skip to hit fps_target
        # e.g. src=25, target=5 → process every 5th frame
        skip_interval = max(1, int(round(fps_src / fps_target)))

        # Output video writer
        writer = None
        if save_output:
            Path("outputs/videos").mkdir(parents=True, exist_ok=True)
            out_path = str(Path("outputs/videos") / (Path(video_path).stem + "_annotated.mp4"))
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps_target, (640, 480))

        frame_id      = 0   # source frame counter
        processed_id  = 0   # frames actually analysed
        t_prev        = time.time()
        t_start       = t_prev
        results       = []

        effective_total = (total // skip_interval) if total > 0 else "?"
        print(f"\n[START] Processing: {video_path}")
        print(f"   Source frames: {total} @ {fps_src:.1f} fps")
        print(f"   Skip interval: {skip_interval}  -> ~{effective_total} frames to analyse")
        print(f"   Anomaly threshold: {threshold}")
        print(f"   YOLO every {yolo_every_n_frames} analysed frames\n")

        _last_obj_frame = None   # reuse last YOLO result between subsampled calls
        anomaly_score   = 0.0    # cached; updated every seq_len processed frames
        
        prev_frame_gray = None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Frame-skip: only analyse every skip_interval-th frame ─────
            if frame_id % skip_interval != 0:
                frame_id += 1
                continue

            if max_frames and processed_id >= max_frames:
                break

            # ── Motion Energy & Enhancement ───────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_enhanced = clahe.apply(gray)
            frame_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
            
            motion_score = 0.0
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray_enhanced, prev_frame_gray)
                _, thresh_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_score = float(np.sum(thresh_diff) / (gray.shape[0] * gray.shape[1]))
            prev_frame_gray = gray_enhanced
            
            # MANUAL FIX 1: Gaussian Blur to reduce sensor noise
            blurred = cv2.GaussianBlur(frame_enhanced, (5, 5), 0)

            # ── Module 2: Heatmap ─────────────────────────────────────────
            need_overlay = bool(writer or display)
            heatmap_frame, density_score = self.heatmap.process_frame(blurred, return_overlay=need_overlay)

            # ── Module 1: Anomaly Detection (Sliding Window) ──────────────
            tensor = self.preprocess_frame(blurred)
            self.frame_buffer.append(tensor)
            
            # Calculate raw reconstruction error
            raw_err = self.get_anomaly_score()

            # ── Module 3: Abandoned Object Detection (subsampled) ─────────
            # Resize frame to 320×240 before YOLO to reduce inference time.
            # Run YOLO only every yolo_every_n_frames processed frames.
            yolo_boost = 0.0
            if processed_id % yolo_every_n_frames == 0:
                small = cv2.resize(blurred, (320, 240))
                _last_obj_frame, abandoned_events = self.obj_tracker.process_frame(small, frame_id)
            else:
                # Reuse last annotated frame; still update stationary timers
                abandoned_events = []
                _last_obj_frame = _last_obj_frame if _last_obj_frame is not None else blurred

            for t in self.obj_tracker.get_track_summary():
                if t['class_name'] in ['bicycle', 'motorcycle', 'truck', 'bus', 'car', 'skateboard']:
                    yolo_boost = 1.0

            # ── Blend and Smooth Scores ───────────────────────────────────
            blended_score = max(raw_err, yolo_boost, motion_score * 5.0)
            self.score_buffer.append(blended_score)
            anomaly_score = float(np.mean(list(self.score_buffer))) if len(self.score_buffer) > 0 else 0.0
            
            is_anomaly = anomaly_score > threshold

            for ev in abandoned_events:
                self.alerts.check_abandoned_object(
                    frame_id=ev["frame_id"],
                    object_id=ev["obj_id"],
                    stationary_seconds=ev["stationary_seconds"],
                )
                print(f"[ABANDONED] {ev['class_name']} stationary "
                      f"{ev['stationary_seconds']:.0f}s @ frame {ev['frame_id']}")

            # ── Alert Engine: anomaly + density ───────────────────────────
            self.alerts.check_anomaly(frame_id, anomaly_score)
            self.alerts.check_density(frame_id, density_score)

            # ── FPS calculation ───────────────────────────────────────────
            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            self.alerts.log_metrics(fps, density_score, anomaly_score, frame_id)

            # ── Annotation ────────────────────────────────────────────────
            if need_overlay:
                obj_resized     = cv2.resize(_last_obj_frame, (640, 480))
                heatmap_resized = cv2.resize(heatmap_frame, (640, 480))
                display_frame   = cv2.addWeighted(heatmap_resized, 0.7, obj_resized, 0.3, 0)

                color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                label = f"{'ANOMALY' if is_anomaly else 'NORMAL'}  Score:{anomaly_score:.2f}"
                n_abandoned = len([o for o in self.obj_tracker.get_track_summary() if o['alerted']])
                cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 0), -1)
                cv2.putText(display_frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(
                    display_frame,
                    f"Density:{density_score:.2f}  FPS:{fps:.1f}  Abandoned:{n_abandoned}  Frame:{frame_id}",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
                )

            results.append({
                "frame_id": frame_id,
                "anomaly_score": round(anomaly_score, 4),
                "density_score": round(density_score, 4),
                "is_anomaly": is_anomaly,
            })

            if writer:
                writer.write(display_frame)
            if display:
                cv2.imshow("Crowd Anomaly Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Progress every 10 processed frames so UI stays live ───────
            if processed_id % 10 == 0 and processed_id > 0:
                elapsed = time.time() - t_start
                pct = (frame_id / total * 100) if total > 0 else 0
                print(f"   [Progress] frame {frame_id}/{total} ({pct:.0f}%)  "
                      f"processed={processed_id}  elapsed={elapsed:.0f}s  fps={fps:.1f}")
                if progress_callback:
                    progress_callback(processed_id, total // skip_interval or processed_id)

            frame_id     += 1
            processed_id += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self.obj_tracker.reset()
        self.heatmap.reset()

        # Save JSON results
        os.makedirs("outputs", exist_ok=True)
        result_path = f"outputs/{Path(video_path).stem}_results.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[DONE] {frame_id} frames processed")
        print(f"   Results saved -> {result_path}")
        if save_output:
            print(f"   Annotated video -> outputs/videos/{Path(video_path).stem}_annotated.mp4")
        return results


def main():
    args = parse_args()
    pipeline = InferencePipeline(config_path=args.config)
    pipeline.run(
        video_path=args.video,
        threshold=args.threshold,
        save_output=args.save,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
