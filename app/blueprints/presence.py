# app/blueprints/presence.py
from datetime import datetime, timezone
from flask import Blueprint, jsonify
from flask_login import login_required, current_user
from app import db
from app.utils.presence import mark_online

bp = Blueprint("presence", __name__, url_prefix="/presence")

# ---- CSRF免除（Flask-WTF使用時）----
try:
    from app import csrf  # app/__init__.py で CSRFProtect(app) を csfr として保持している想定
    csrf.exempt(bp)
except Exception:
    # CSRF未導入環境なら何もしない
    pass

_DB_UPDATE_COOLDOWN_SEC = 60

@bp.route("/ping", methods=["POST"])
@login_required
def ping():
    mark_online(current_user.id)
    now = datetime.now(timezone.utc)
    if (current_user.last_seen_at is None) or ((now - current_user.last_seen_at).total_seconds() > _DB_UPDATE_COOLDOWN_SEC):
        current_user.last_seen_at = now
        db.session.commit()
    return jsonify(ok=True)
