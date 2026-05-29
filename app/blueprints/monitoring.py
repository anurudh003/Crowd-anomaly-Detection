from flask import Blueprint, render_template, jsonify, current_app, request, Response
from .auth import login_required
import psutil
import time

try:
    import GPUtil
except ImportError:
    GPUtil = None

monitoring_bp = Blueprint('monitoring', __name__)

@monitoring_bp.route("/monitoring")
@login_required
def index():
    return render_template("modules/monitoring.html")

@monitoring_bp.route("/api/anomalies")
@login_required
def api_anomalies():
    limit = int(request.args.get("limit", 50))
    return jsonify(current_app.alert_engine.get_recent_alerts(limit))

@monitoring_bp.route("/api/alerts/<int:alert_id>/dismiss", methods=["POST", "DELETE"])
@login_required
def dismiss_alert(alert_id):
    current_app.alert_engine.dismiss_alert(alert_id)
    return jsonify({"success": True, "id": alert_id})

@monitoring_bp.route("/api/alerts/dismiss_all", methods=["POST"])
@login_required
def dismiss_all_alerts():
    """Dismiss all current alerts."""
    import sqlite3
    db_path = current_app.alert_engine.db_path
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM alerts")
        conn.commit()
    return jsonify({"success": True})

@monitoring_bp.route("/api/metrics")
@login_required
def api_metrics():
    limit = int(request.args.get("limit", 100))
    return jsonify(current_app.alert_engine.get_metrics(limit))

@monitoring_bp.route("/api/hardware")
@login_required
def api_hardware():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    mem_used_gb = round(mem.used / 1e9, 2)
    mem_total_gb = round(mem.total / 1e9, 2)

    # Disk info
    try:
        disk = psutil.disk_usage('/')
        disk_pct = disk.percent
    except Exception:
        disk_pct = 0

    # Network I/O
    try:
        net = psutil.net_io_counters()
        net_sent_mb = round(net.bytes_sent / 1e6, 1)
        net_recv_mb = round(net.bytes_recv / 1e6, 1)
    except Exception:
        net_sent_mb = net_recv_mb = 0

    gpu_percent = 0
    gpu_name = "N/A"
    gpu_mem_pct = 0
    if GPUtil:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = round(gpus[0].load * 100, 1)
            gpu_name = gpus[0].name
            gpu_mem_pct = round(gpus[0].memoryUtil * 100, 1)

    # Process-level stats
    try:
        proc = psutil.Process()
        proc_cpu = round(proc.cpu_percent(interval=None), 1)
        proc_mem_mb = round(proc.memory_info().rss / 1e6, 1)
        proc_threads = proc.num_threads()
    except Exception:
        proc_cpu = proc_mem_mb = proc_threads = 0

    return jsonify({
        "cpu": cpu_percent,
        "mem": mem_percent,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "disk_pct": disk_pct,
        "net_sent_mb": net_sent_mb,
        "net_recv_mb": net_recv_mb,
        "gpu": gpu_percent,
        "gpu_name": gpu_name,
        "gpu_mem_pct": gpu_mem_pct,
        "proc_cpu": proc_cpu,
        "proc_mem_mb": proc_mem_mb,
        "proc_threads": proc_threads,
        "latency": round(1000.0 / max(cpu_percent, 1), 1),  # Estimated
    })


def _generate_frames(frame_getter, fallback_path=None):
    """Generic MJPEG frame generator."""
    while True:
        frame = frame_getter()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)


def generate_frames():
    from ..core.state import global_state
    while True:
        frame = global_state.get_snapshot().get("latest_frame")
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)


@monitoring_bp.route("/api/stream")
@login_required
def api_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
