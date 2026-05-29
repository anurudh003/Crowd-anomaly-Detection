import json
import os
from pathlib import Path
from app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # Resolve config relative to this file's directory (project root)
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    dash = cfg["dashboard"]
    print(f"\n[INFO] Urban Sentinel AI Dashboard running at http://{dash['host']}:{dash['port']}")
    socketio.run(
        app,
        host=dash["host"],
        port=dash["port"],
        debug=dash["debug"],
        allow_unsafe_werkzeug=True,
    )
