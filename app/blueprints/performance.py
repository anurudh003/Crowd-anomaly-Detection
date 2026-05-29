import os
import threading
import time
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

from ..core.pipeline import InferencePipeline
from ..core.state import global_state
from .auth import login_required

performance_bp = Blueprint('performance', __name__)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@performance_bp.route("/performance")
@login_required
def index():
    return render_template("modules/performance.html")

@performance_bp.route("/api/process_video", methods=["POST"])
@login_required
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    if global_state.get_snapshot()["running"]:
        return jsonify({"error": "Already processing a video. Please wait."}), 429

    filename = secure_filename(file.filename)
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    threshold = float(request.form.get("threshold", 0.31))
    app_obj = current_app._get_current_object()

    def process_in_background():
        with app_obj.app_context():
            import cv2
            cap_peek = cv2.VideoCapture(save_path)
            total = int(cap_peek.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_peek.release()

            global_state.reset(filename=filename, total_frames=total)
            global_state.update(threshold=threshold)

            # Start session in DB
            session_id = app_obj.alert_engine.start_session(filename=filename)

            def on_progress(frames_done: int, total_frames: int):
                elapsed = time.time() - global_state.get_snapshot()["started_at"]
                pct = round(frames_done / total_frames * 100, 1) if total_frames > 0 else 0
                global_state.update(
                    frames_done=frames_done,
                    total_frames=total_frames,
                    pct_complete=pct,
                    elapsed_seconds=round(elapsed, 1)
                )

            try:
                # Clear previous session data for fresh start
                app_obj.alert_engine.clear_metrics()

                pipeline = InferencePipeline(alert_engine=app_obj.alert_engine)
                results = pipeline.run(
                    video_path=save_path,
                    threshold=threshold,
                    save_output=False,
                    display=False,
                    progress_callback=on_progress,
                    session_id=session_id,
                )
                global_state.update(result=results, pct_complete=100)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ERROR] Pipeline failed: {e}\n{tb}")
                global_state.update(error=str(e))
            finally:
                elapsed = time.time() - global_state.get_snapshot()["started_at"]
                global_state.update(running=False, elapsed_seconds=round(elapsed, 1))

    t = threading.Thread(target=process_in_background, daemon=True)
    t.start()

    return jsonify({
        "message": f"Processing started for {filename}",
        "filename": filename,
        "threshold": threshold,
    })

@performance_bp.route("/api/status")
@login_required
def status():
    snapshot = global_state.get_snapshot()
    result_summary = None
    if snapshot["result"] is not None:
        frames = snapshot["result"]
        if isinstance(frames, list) and len(frames) > 0:
            n_anomaly = sum(1 for r in frames if r.get("is_anomaly"))
            avg_score = sum(r.get("anomaly_score", 0) for r in frames) / len(frames)
            avg_density = sum(r.get("density_score", 0) for r in frames) / len(frames)
            result_summary = {
                "total_frames":   len(frames),
                "anomaly_frames": n_anomaly,
                "anomaly_rate":   round(n_anomaly / len(frames) * 100, 1) if frames else 0,
                "avg_anomaly_score": round(avg_score, 4),
                "avg_density_score": round(avg_density, 4),
            }
        else:
            result_summary = frames
    snapshot["result_summary"] = result_summary
    snapshot.pop("result", None)
    return jsonify(snapshot)

@performance_bp.route("/api/sessions")
@login_required
def api_sessions():
    """Return past processing sessions."""
    sessions = current_app.alert_engine.get_sessions(limit=10)
    return jsonify(sessions)

@performance_bp.route("/api/diagnostics")
@login_required
def api_diagnostics():
    """Run system diagnostics."""
    result = current_app.alert_engine.get_diagnostics()
    return jsonify(result)
