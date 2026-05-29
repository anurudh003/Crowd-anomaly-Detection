from flask import Blueprint, render_template, jsonify, current_app, request, Response
from .auth import login_required
import time

heatmaps_bp = Blueprint('heatmaps', __name__)

@heatmaps_bp.route('/heatmaps')
@login_required
def index():
    return render_template('modules/heatmaps.html')

@heatmaps_bp.route('/api/heatmap_stream')
@login_required
def heatmap_stream():
    """MJPEG stream of the real-time heatmap frame from the active pipeline."""
    def generate():
        from ..core.state import global_state
        # Access the pipeline's heatmap frame via global state
        while True:
            frame = global_state.get_snapshot().get("latest_heatmap_frame")
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
