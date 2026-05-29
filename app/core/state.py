import threading
import time

class ProcessingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "running": False,
            "filename": None,
            "frames_done": 0,
            "total_frames": 0,
            "pct_complete": 0,
            "elapsed_seconds": 0,
            "started_at": None,
            "result": None,
            "error": None,
            "latest_frame": None,           # Composite display frame (MJPEG stream)
            "latest_heatmap_frame": None,   # Pure heatmap overlay (heatmap stream)
            "latest_detection_frame": None, # YOLO annotated frame (detection stream)
            "threshold": 0.31,
        }

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if key in self.state:
                    self.state[key] = value

    def get_snapshot(self):
        with self.lock:
            return dict(self.state)

    def reset(self, filename=None, total_frames=0):
        with self.lock:
            self.state["running"] = True
            self.state["filename"] = filename
            self.state["frames_done"] = 0
            self.state["total_frames"] = total_frames
            self.state["pct_complete"] = 0
            self.state["elapsed_seconds"] = 0
            self.state["started_at"] = time.time()
            self.state["result"] = None
            self.state["error"] = None
            self.state["latest_frame"] = None
            self.state["latest_heatmap_frame"] = None
            self.state["latest_detection_frame"] = None

global_state = ProcessingState()
