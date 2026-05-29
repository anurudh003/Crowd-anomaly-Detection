from flask import Blueprint, render_template
from .auth import login_required

architecture_bp = Blueprint('architecture', __name__)

@architecture_bp.route('/architecture')
@login_required
def index():
    return render_template('modules/architecture.html')

