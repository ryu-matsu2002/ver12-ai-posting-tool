# app/controllers/external_seo.py
from flask import Blueprint, render_template, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app.models import Site
from app.tasks import enqueue_external_seo

external_bp = Blueprint("external", __name__, url_prefix="/external")

@external_bp.route("/sites")
@login_required
def external_sites():
    sites = (Site.query.filter_by(user_id=current_user.id)
             if not current_user.is_admin else Site.query.all())
    return render_template("external_sites.html", sites=sites)

@external_bp.route("/start/<int:site_id>", methods=["POST"])
@login_required
def start_external(site_id):
    enqueue_external_seo.delay(site_id)   # Celery or APScheduler wrapper
    return jsonify({"status": "queued"})
