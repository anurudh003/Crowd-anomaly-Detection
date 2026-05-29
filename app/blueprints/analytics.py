from flask import Blueprint, render_template
from .auth import login_required

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics')
@login_required
def index():
    return render_template('modules/analytics.html')

