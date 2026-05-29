from flask import Blueprint, render_template
from .auth import login_required

detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/detection')
@login_required
def index():
    return render_template('modules/detection.html')

