"""
Crowd Anomaly Detection - Alert Engine
Handles: alert generation, debouncing, severity scoring, SQLite logging
"""

import sqlite3
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    timestamp: str
    alert_type: str          # "ANOMALY" | "OVERCROWDING" | "ABANDONED_OBJECT"
    severity: str            # Severity enum value
    frame_id: int
    anomaly_score: float
    details: str
    snapshot_b64: Optional[str] = field(default=None)   # Base64 frame snapshot
    id: Optional[int] = field(default=None)

    def to_dict(self) -> Dict:
        return asdict(self)


class AlertEngine:
    """
    Centralized alert engine that:
      - Receives signals from all 3 modules
      - Debounces duplicate alerts within a cooldown window
      - Assigns severity levels
      - Persists alerts to SQLite
      - Provides REST-ready JSON for the dashboard
    """

    SEVERITY_THRESHOLDS = {
        (0.0, 0.4): Severity.LOW,
        (0.4, 0.6): Severity.MEDIUM,
        (0.6, 0.8): Severity.HIGH,
        (0.8, 1.0): Severity.CRITICAL,
    }

    def __init__(self, config_path: str = "config.json"):
        # Resolve config path relative to project root (works from any CWD)
        if not os.path.isabs(config_path):
            root = Path(__file__).parent.parent
            config_path = str(root / config_path)

        with open(config_path) as f:
            cfg = json.load(f)

        alert_cfg = cfg["alerts"]
        self.debounce_sec       = alert_cfg.get("debounce_seconds", 30)
        self.density_threshold  = alert_cfg.get("density_crowd_threshold", 0.75)
        self.anomaly_threshold  = cfg.get("inference", {}).get("anomaly_threshold", 0.5)
        db_path                 = alert_cfg.get("log_path", "logs/alerts.db")

        # Resolve db path relative to project root
        if not os.path.isabs(db_path):
            root = Path(__file__).parent.parent
            db_path = str(root / db_path)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

        # Debounce tracker: alert_type → last_triggered_time
        self._last_alert_time: Dict[str, float] = {}

        # Diagnostics tracking
        self._session_start = time.time()
        self._frames_processed = 0

    # ── Database ────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create SQLite schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    alert_type    TEXT    NOT NULL,
                    severity      TEXT    NOT NULL,
                    frame_id      INTEGER NOT NULL,
                    anomaly_score REAL    NOT NULL,
                    details       TEXT,
                    snapshot_b64  TEXT
                )
            """)
            # Add snapshot_b64 column if upgrading from older schema
            try:
                conn.execute("ALTER TABLE alerts ADD COLUMN snapshot_b64 TEXT")
            except Exception:
                pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT    NOT NULL,
                    fps            REAL,
                    density_score  REAL,
                    anomaly_score  REAL,
                    frames_processed INTEGER,
                    current_threshold REAL
                )
            """)
            # Add current_threshold if upgrading
            try:
                conn.execute("ALTER TABLE system_metrics ADD COLUMN current_threshold REAL")
            except Exception:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at   TEXT NOT NULL,
                    ended_at     TEXT,
                    filename     TEXT,
                    total_frames INTEGER DEFAULT 0,
                    anomaly_frames INTEGER DEFAULT 0,
                    anomaly_rate REAL DEFAULT 0,
                    status       TEXT DEFAULT 'running'
                )
            """)
            conn.commit()

    def _save_alert(self, alert: Alert):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, frame_id, anomaly_score, details, snapshot_b64)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert.timestamp, alert.alert_type, alert.severity,
                  alert.frame_id, alert.anomaly_score, alert.details,
                  alert.snapshot_b64))
            conn.commit()
            alert.id = cur.lastrowid

    def log_metrics(self, fps: float, density_score: float, anomaly_score: float,
                    frames: int, current_threshold: float = None):
        """Log system performance and detection metrics to SQLite."""
        self._frames_processed = frames
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    INSERT INTO system_metrics
                    (timestamp, fps, density_score, anomaly_score, frames_processed, current_threshold)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.utcnow().isoformat(), fps, density_score,
                      anomaly_score, frames, current_threshold))
                conn.commit()
        except Exception as e:
            print(f"[WARN] Metric log failed: {e}")

    def clear_metrics(self):
        """Clear all metrics and alerts for a fresh session."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("DELETE FROM system_metrics")
                conn.execute("DELETE FROM alerts")
                conn.commit()
            self._session_start = time.time()
            self._frames_processed = 0
            print("[INFO] Performance metrics and alert history cleared for new session.")
        except Exception as e:
            print(f"[WARN] Failed to clear metrics: {e}")

    def start_session(self, filename: str = None):
        """Record session start."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cur = conn.execute("""
                    INSERT INTO sessions (started_at, filename, status)
                    VALUES (?, ?, 'running')
                """, (datetime.utcnow().isoformat(), filename))
                conn.commit()
                return cur.lastrowid
        except Exception:
            return None

    def end_session(self, session_id: int, total_frames: int, anomaly_frames: int):
        """Record session end with stats."""
        if not session_id:
            return
        rate = round(anomaly_frames / total_frames * 100, 1) if total_frames > 0 else 0
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    UPDATE sessions
                    SET ended_at=?, total_frames=?, anomaly_frames=?, anomaly_rate=?, status='complete'
                    WHERE id=?
                """, (datetime.utcnow().isoformat(), total_frames, anomaly_frames, rate, session_id))
                conn.commit()
        except Exception:
            pass

    def get_sessions(self, limit: int = 10) -> List[Dict]:
        """Get past processing sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ── Alert Generation ─────────────────────────────────────────────────────

    def _get_severity(self, score: float) -> Severity:
        for (lo, hi), sev in self.SEVERITY_THRESHOLDS.items():
            if lo <= score < hi:
                return sev
        return Severity.CRITICAL

    def _is_debounced(self, alert_type: str) -> bool:
        last = self._last_alert_time.get(alert_type, 0)
        return (time.time() - last) < self.debounce_sec

    def _trigger(self, alert: Alert) -> Optional[Alert]:
        if self._is_debounced(alert.alert_type):
            return None
        self._last_alert_time[alert.alert_type] = time.time()
        self._save_alert(alert)
        print(f"[ALERT] {alert.severity} {alert.alert_type} @ frame {alert.frame_id} "
              f"(score={alert.anomaly_score:.3f})")
        return alert

    def check_anomaly(self, frame_id: int, anomaly_score: float,
                      dynamic_threshold: float = None,
                      frame_snapshot_b64: str = None) -> Optional[Alert]:
        """Call when anomaly detector outputs a score."""
        threshold_to_use = dynamic_threshold if dynamic_threshold is not None else self.anomaly_threshold
        if anomaly_score < threshold_to_use:
            return None
        alert = Alert(
            timestamp     = datetime.utcnow().isoformat(),
            alert_type    = "ANOMALY",
            severity      = self._get_severity(anomaly_score).value,
            frame_id      = frame_id,
            anomaly_score = anomaly_score,
            details       = f"Anomalous crowd behavior detected (score={anomaly_score:.3f})",
            snapshot_b64  = frame_snapshot_b64,
        )
        return self._trigger(alert)

    def check_density(self, frame_id: int, density_score: float,
                      frame_snapshot_b64: str = None) -> Optional[Alert]:
        """Call when heatmap module outputs a density score."""
        if density_score < self.density_threshold:
            return None
        alert = Alert(
            timestamp     = datetime.utcnow().isoformat(),
            alert_type    = "OVERCROWDING",
            severity      = self._get_severity(density_score).value,
            frame_id      = frame_id,
            anomaly_score = density_score,
            details       = f"Crowd density exceeded threshold (density={density_score:.3f})",
            snapshot_b64  = frame_snapshot_b64,
        )
        return self._trigger(alert)

    def check_abandoned_object(self, frame_id: int, object_id: str,
                                stationary_seconds: float,
                                class_name: str = "object",
                                snapshot_b64: str = None) -> Optional[Alert]:
        """Call when object tracker detects stationary object."""
        key = f"ABANDONED_{object_id}"
        score = min(1.0, stationary_seconds / 60.0)
        alert = Alert(
            timestamp     = datetime.utcnow().isoformat(),
            alert_type    = "ABANDONED_OBJECT",
            severity      = self._get_severity(score).value,
            frame_id      = frame_id,
            anomaly_score = score,
            details       = f"'{class_name}' [{object_id}] stationary for {stationary_seconds:.0f}s",
            snapshot_b64  = snapshot_b64,
        )
        if self._is_debounced(key):
            return None
        self._last_alert_time[key] = time.time()
        self._save_alert(alert)
        return alert

    # ── Dashboard API ─────────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Fetch most recent alerts for REST API."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def dismiss_alert(self, alert_id: int):
        """Dismiss an alert by deleting it from the active database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            conn.commit()

    def get_metrics(self, limit: int = 100) -> List[Dict]:
        """Fetch recent system metrics for chart display."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM system_metrics ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_diagnostics(self) -> Dict:
        """Run a full system diagnostic and return results."""
        import platform
        import sys
        diag = {
            "status": "OK",
            "checks": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        issues = []

        # 1. DB connectivity
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
            diag["checks"].append({"name": "SQLite Database", "status": "PASS",
                                   "detail": f"Connected: {self.db_path}"})
        except Exception as e:
            diag["checks"].append({"name": "SQLite Database", "status": "FAIL",
                                   "detail": str(e)})
            issues.append("DB connection failed")

        # 2. Model file
        root = Path(self.db_path).parent.parent
        model_path = root / "models" / "anomaly_detector.pth"
        if model_path.exists():
            size_mb = round(model_path.stat().st_size / 1e6, 1)
            diag["checks"].append({"name": "Anomaly Model", "status": "PASS",
                                   "detail": f"Found ({size_mb} MB)"})
        else:
            diag["checks"].append({"name": "Anomaly Model", "status": "WARN",
                                   "detail": "Not found – running in demo mode (random weights)"})

        # 3. YOLO model
        yolo_path = root / "yolov5nu.pt"
        if yolo_path.exists():
            size_mb = round(yolo_path.stat().st_size / 1e6, 1)
            diag["checks"].append({"name": "YOLO v5n Model", "status": "PASS",
                                   "detail": f"Found ({size_mb} MB)"})
        else:
            diag["checks"].append({"name": "YOLO v5n Model", "status": "FAIL",
                                   "detail": "yolov5nu.pt not found"})
            issues.append("YOLO model missing")

        # 4. PyTorch
        try:
            import torch
            diag["checks"].append({"name": "PyTorch", "status": "PASS",
                                   "detail": f"v{torch.__version__} | CUDA: {torch.cuda.is_available()}"})
        except ImportError:
            diag["checks"].append({"name": "PyTorch", "status": "FAIL", "detail": "Not installed"})
            issues.append("PyTorch missing")

        # 5. OpenCV
        try:
            import cv2
            diag["checks"].append({"name": "OpenCV", "status": "PASS",
                                   "detail": f"v{cv2.__version__}"})
        except ImportError:
            diag["checks"].append({"name": "OpenCV", "status": "FAIL", "detail": "Not installed"})
            issues.append("OpenCV missing")

        # 6. Ultralytics
        try:
            import ultralytics
            diag["checks"].append({"name": "Ultralytics (YOLO)", "status": "PASS",
                                   "detail": f"v{ultralytics.__version__}"})
        except ImportError:
            diag["checks"].append({"name": "Ultralytics (YOLO)", "status": "WARN",
                                   "detail": "Not installed – object tracking disabled"})

        # 7. Upload folder
        uploads = root / "data" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        diag["checks"].append({"name": "Upload Folder", "status": "PASS",
                               "detail": str(uploads)})

        # 8. Logs folder
        logs_dir = Path(self.db_path).parent
        diag["checks"].append({"name": "Logs / DB Folder", "status": "PASS",
                               "detail": str(logs_dir)})

        # 9. Python & Platform
        diag["checks"].append({"name": "Python Runtime", "status": "PASS",
                               "detail": f"{sys.version.split()[0]} | {platform.system()} {platform.release()}"})

        # 10. Metric counts
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                n_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
                n_metrics = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
            diag["checks"].append({"name": "Stored Data", "status": "PASS",
                                   "detail": f"{n_alerts} alerts · {n_metrics} metric records"})
        except Exception:
            pass

        diag["status"] = "ISSUES_FOUND" if issues else "OK"
        diag["issues"] = issues
        return diag


if __name__ == "__main__":
    engine = AlertEngine()
    engine.check_anomaly(frame_id=100, anomaly_score=0.85)
    engine.check_density(frame_id=101, density_score=0.80)
    print("Recent alerts:", engine.get_recent_alerts(5))
    print("Diagnostics:", engine.get_diagnostics())
    print("✅ Alert engine OK")
