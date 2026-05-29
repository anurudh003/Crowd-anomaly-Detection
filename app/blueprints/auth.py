from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
import time
from app import cache
from functools import wraps

auth_bp = Blueprint('auth', __name__)

USERS = {
    "DIR-001": {"passkey": "director_auth", "role": "Director"},
    "ANA-492": {"passkey": "analyst_auth", "role": "Data Analyst"},
    "OPS-773": {"passkey": "operator_auth", "role": "Operator"}
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "agent_id" not in session:
            return redirect(url_for('auth.login', error="Unauthorized. Please authenticate first."))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route("/")
@cache.cached(timeout=300)
def landing():
    return render_template("auth/landing.html")

@auth_bp.before_app_request
def enforce_login():
    # List of endpoints that don't require login
    exempt_endpoints = ['auth.login', 'auth.landing', 'static']
    
    if "agent_id" not in session and request.endpoint not in exempt_endpoints:
        if request.endpoint: # Only redirect if it's a valid endpoint
             return redirect(url_for('auth.login', error="Unauthorized. Please authenticate first."))

@auth_bp.route("/login", methods=["GET", "POST"])
@cache.cached(timeout=60, query_string=True, unless=lambda: request.method == 'POST')
def login():
    error = request.args.get("error")
    if request.method == "POST":
        agent_id = request.form.get("agent_id")
        passkey = request.form.get("passkey")
        
        if agent_id in USERS and USERS[agent_id]["passkey"] == passkey:
            session["user_role"] = USERS[agent_id]["role"]
            session["agent_id"] = agent_id
            return redirect(url_for("monitoring.index"))
        else:
            time.sleep(1)  # Security delay
            error = "Authentication Failed. Invalid Agent ID or Passkey."
            
    return render_template("auth/login.html", error=error)

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
