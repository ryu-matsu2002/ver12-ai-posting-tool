# app/blueprints/presence.py
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, make_response
from flask_login import login_required, current_user
from app import db
from app.utils.presence import mark_online, online_id_set
from app.models import User

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
    # Redis失敗時の挙動は運用方針に合わせて：ここではエラーを握りつぶして200返す
    try:
        mark_online(current_user.id)
    except Exception:
        pass
    now = datetime.now(timezone.utc)
    if (current_user.last_seen_at is None) or ((now - current_user.last_seen_at).total_seconds() > _DB_UPDATE_COOLDOWN_SEC):
        current_user.last_seen_at = now
        db.session.commit()
    resp = make_response(jsonify(ok=True))
    resp.headers["Cache-Control"] = "no-store"
    return resp



@bp.route("/status", methods=["GET"])
@login_required
def status_batch():
    ids = request.args.get("user_ids", "").strip()
    try:
        # 重複除去 & 上限ガード（100件）
        user_ids = list({int(x) for x in ids.split(",") if x})[:100]
    except ValueError:
        return jsonify(error="bad user_ids"), 400
    if not user_ids:
        resp = make_response(jsonify(online_ids=[], last_seen_at={}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # 表示許可ユーザーだけ対象
    q = User.query.filter(User.id.in_(user_ids), User.share_presence.is_(True))
    # 多テナントなら以下のような境界を追加（例）
    # q = q.filter(User.tenant_id == current_user.tenant_id)
    users = q.all()
    visible_ids = [u.id for u in users]

    try:
        online_ids = list(online_id_set(visible_ids))
    except Exception:
        # Redis失敗時は空集合で返す
        online_ids = []
    last_seen = {str(u.id): (u.last_seen_at.isoformat() if u.last_seen_at else None) for u in users}
    resp = make_response(jsonify(online_ids=online_ids, last_seen_at=last_seen))
    resp.headers["Cache-Control"] = "no-store"
    return resp
