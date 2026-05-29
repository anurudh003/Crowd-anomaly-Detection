import os
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO
from flask_caching import Cache
from utils.alert_engine import AlertEngine

# async_mode="threading" is critical: it lets Flask handle MJPEG streams
# AND regular API requests concurrently without connection resets.
socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")
cache = Cache()

def create_app():
    # Resolve project root (the crowd/ directory) — works from any CWD
    project_root = Path(__file__).parent.parent
    config_path  = str(project_root / "config.json")

    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")

    app.secret_key = 'super_secret_cyber_key'

    # Use absolute path for uploads so it's always consistent
    upload_folder = str(project_root / "data" / "uploads")
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

    os.makedirs(upload_folder, exist_ok=True)

    # Initialize Alert Engine with absolute config path
    app.alert_engine = AlertEngine(config_path=config_path)

    # Register Blueprints
    from .blueprints.auth import auth_bp
    from .blueprints.monitoring import monitoring_bp
    from .blueprints.analytics import analytics_bp
    from .blueprints.heatmaps import heatmaps_bp
    from .blueprints.detection import detection_bp
    from .blueprints.performance import performance_bp
    from .blueprints.architecture import architecture_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(heatmaps_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(performance_bp)
    app.register_blueprint(architecture_bp)

    cache.init_app(app, config={'CACHE_TYPE': 'simple'})
    socketio.init_app(app)
    return app
