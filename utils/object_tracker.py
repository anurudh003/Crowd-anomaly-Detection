"""
Crowd Anomaly Detection - Module 3: Abandoned Object Detection
Algorithm:
  1. YOLOv5n detects target objects (bags, backpacks, suitcases) each frame
  2. Centroid tracker matches detections across frames via IoU / distance
  3. Objects stationary for > stationary_time_seconds trigger an alert
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    """Tracks a single detected object across frames."""
    obj_id: str
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    total_movement: float = 0.0       # cumulative pixel movement
    frames_tracked: int = 1
    alerted: bool = False             # True once an alert has been fired

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def stationary_seconds(self) -> float:
        return time.time() - self.first_seen

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update position and accumulate movement."""
        new_cx, new_cy = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        dx = new_cx - self.last_centroid[0]
        dy = new_cy - self.last_centroid[1]
        self.total_movement += (dx ** 2 + dy ** 2) ** 0.5
        self.last_centroid = (new_cx, new_cy)
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_tracked += 1


# ── Main Tracker Class ────────────────────────────────────────────────────────

class AbandonedObjectTracker:
    """
    Module 3: Detects abandoned objects (bags, suitcases, backpacks).

    Detection pipeline:
        frame → YOLOv5n → bounding boxes → centroid tracker
              → check stationary time & movement → alert engine callback
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        ot_cfg = cfg.get("object_tracking", {})
        self.model_name            = ot_cfg.get("model", "yolov5n")
        self.stationary_px         = ot_cfg.get("stationary_pixel_threshold", 20)
        self.stationary_time_sec   = ot_cfg.get("stationary_time_seconds", 30)
        self.conf_threshold        = ot_cfg.get("confidence_threshold", 0.4)
        self.target_classes        = set(ot_cfg.get("target_classes", ["backpack", "suitcase", "handbag"]))

        # YOLO model (lazy-loaded on first process_frame call)
        self._model = None
        self._model_load_error: Optional[str] = None

        # Active tracked objects: obj_id → TrackedObject
        self._objects: Dict[str, TrackedObject] = {}
        self._next_id: int = 0

        # IoU / distance threshold for matching detections to tracks
        self._iou_threshold   = 0.30
        self._max_dist_px     = 80.0
        self._max_absent_sec  = 5.0    # remove track if unseen for this long

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load YOLO model via ultralytics package (first call only)."""
        if self._model is not None or self._model_load_error:
            return
        try:
            from ultralytics import YOLO
            import torch
            # Patch torch.load to avoid weights_only error with older ultralytics
            original_load = torch.load
            def patched_load(*args, **kwargs):
                if 'weights_only' in kwargs:
                    kwargs['weights_only'] = False
                else:
                    kwargs.setdefault('weights_only', False)
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            # Map yolov5n -> yolov5nu.pt for the ultralytics package
            model_file = self.model_name + "u.pt" if "v5" in self.model_name else self.model_name + ".pt"
            self._model = YOLO(model_file)
            torch.load = original_load  # restore
            
            logger.info(f"[AbandonedObjectTracker] Loaded {self.model_name} OK")
        except Exception as exc:
            self._model_load_error = str(exc)
            logger.warning(
                f"[AbandonedObjectTracker] Could not load {self.model_name}: {exc}. "
                "Module 3 will be skipped."
            )

    # ── IoU Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _centroid_dist(a_bbox: Tuple, b_cx: float, b_cy: float) -> float:
        cx = (a_bbox[0] + a_bbox[2]) / 2.0
        cy = (a_bbox[1] + a_bbox[3]) / 2.0
        return ((cx - b_cx) ** 2 + (cy - b_cy) ** 2) ** 0.5

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], str, float]]:
        """
        Run YOLO on frame and return list of (bbox, class_name, confidence)
        for target classes only.
        """
        if self._model is None:
            return []
        try:
            results = self._model(frame, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = r.names[cls_id]
                    if cls_name in self.target_classes and conf >= self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append(((x1, y1, x2, y2), cls_name, conf))
            return detections
        except Exception as exc:
            logger.debug(f"[AbandonedObjectTracker] Detection error: {exc}")
            return []

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _match_and_update(self, detections: List[Tuple]):
        """
        Greedy IoU-then-distance matching of detections to existing tracks.
        New detections spawn new TrackedObjects.
        """
        matched_ids = set()

        for bbox, cls_name, _conf in detections:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            best_id   = None
            best_score = -1.0

            for obj_id, obj in self._objects.items():
                if obj_id in matched_ids:
                    continue
                iou = self._iou(obj.bbox, bbox)
                if iou > self._iou_threshold and iou > best_score:
                    best_score = iou
                    best_id = obj_id

            # Fall back to centroid distance if IoU match fails
            if best_id is None:
                for obj_id, obj in self._objects.items():
                    if obj_id in matched_ids:
                        continue
                    dist = self._centroid_dist(obj.bbox, cx, cy)
                    if dist < self._max_dist_px and (-dist) > best_score:
                        best_score = -dist
                        best_id = obj_id

            if best_id:
                self._objects[best_id].update(bbox)
                matched_ids.add(best_id)
            else:
                # New object
                new_id = f"obj_{self._next_id:04d}"
                self._next_id += 1
                obj = TrackedObject(
                    obj_id=new_id,
                    bbox=bbox,
                    class_name=cls_name,
                )
                obj.last_centroid = (cx, cy)
                self._objects[new_id] = obj

        # Remove stale tracks
        now = time.time()
        stale = [
            oid for oid, obj in self._objects.items()
            if (now - obj.last_seen) > self._max_absent_sec
        ]
        for oid in stale:
            del self._objects[oid]

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single BGR frame.

        Returns:
            annotated_frame: frame with bounding boxes drawn
            abandoned_events: list of dicts for each newly-abandoned object
                              {obj_id, class_name, stationary_seconds, bbox}
        """
        # Lazy model load (only on first call)
        if self._model is None and self._model_load_error is None:
            self._load_model()

        detections = self._detect(frame)
        self._match_and_update(detections)

        annotated = frame.copy()
        abandoned_events: List[Dict] = []

        for obj in self._objects.values():
            is_stationary = (
                obj.stationary_seconds >= self.stationary_time_sec
                and obj.total_movement < self.stationary_px
            )

            # Draw bounding box
            color = (0, 0, 255) if is_stationary else (0, 165, 255)
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{obj.class_name} {obj.stationary_seconds:.0f}s"
                if is_stationary
                else obj.class_name
            )
            cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Emit event only once per object (alerted flag prevents duplicate alerts)
            if is_stationary and not obj.alerted:
                obj.alerted = True
                abandoned_events.append({
                    "obj_id":             obj.obj_id,
                    "class_name":         obj.class_name,
                    "stationary_seconds": obj.stationary_seconds,
                    "bbox":               list(obj.bbox),
                    "frame_id":           frame_id,
                })

        return annotated, abandoned_events

    def reset(self):
        """Clear all tracks (call between videos)."""
        self._objects.clear()
        self._next_id = 0

    def active_track_count(self) -> int:
        return len(self._objects)

    def get_track_summary(self) -> List[Dict]:
        """Return current track info for debugging."""
        return [
            {
                "obj_id":             o.obj_id,
                "class_name":         o.class_name,
                "stationary_seconds": round(o.stationary_seconds, 1),
                "total_movement_px":  round(o.total_movement, 1),
                "frames_tracked":     o.frames_tracked,
                "alerted":            o.alerted,
            }
            for o in self._objects.values()
        ]


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    tracker = AbandonedObjectTracker()
    cap = cv2.VideoCapture(source)
    frame_id = 0

    print(f"Running abandoned-object tracker on: {source} — press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, events = tracker.process_frame(frame, frame_id)
        for ev in events:
            print(f"[ABANDONED] {ev['class_name']} stationary {ev['stationary_seconds']:.0f}s "
                  f"at frame {ev['frame_id']}")

        cv2.putText(annotated, f"Tracks: {tracker.active_track_count()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Abandoned Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
