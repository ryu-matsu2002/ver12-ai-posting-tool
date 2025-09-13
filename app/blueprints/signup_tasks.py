# app/blueprints/signup_tasks.py
from __future__ import annotations
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, abort
from flask_login import login_required, current_user
from app import db
from app.models import ExternalSignupTask

bp = Blueprint("signup_tasks", __name__, url_prefix="/api/signup/tasks")

TTL_MINUTES = 10  # ワンタイムタスクの有効期限（必要なら環境変数化）

def _now():
    return datetime.utcnow()

def _require_json():
    if not request.is_json:
        abort(400, description="JSON body required")
    return request.get_json(silent=True) or {}

@bp.post("")
@login_required
def create_task():
    """
    タスク生成（ログイン必須）
    入力: {site_id:int, provider:str?=livedoor, payload:object?}
    出力: {token:string, expires_at:string}
    """
    body = _require_json()
    site_id = body.get("site_id")
    if not site_id:
        abort(400, description="site_id required")

    provider = (body.get("provider") or "livedoor").lower().strip()
    payload = body.get("payload") or {}

    tok = __import__("secrets").token_urlsafe(32)
    task = ExternalSignupTask(
        token=tok,
        user_id=current_user.id,
        site_id=int(site_id),
        provider=provider,
        payload=payload,
        status="pending",
        expires_at=_now() + timedelta(minutes=TTL_MINUTES),
    )
    db.session.add(task)
    db.session.commit()

    return jsonify({
        "token": tok,
        "expires_at": task.expires_at.isoformat() + "Z",
        "status": task.status,
    })

@bp.get("/<token>")
def get_task(token: str):
    """
    ヘルパーが取得（ログイン不要：トークン認証）
    出力: {provider, payload, status, expires_at, message}
    """
    task = ExternalSignupTask.query.filter_by(token=token).first()
    if not task:
        abort(404, description="task not found")
    if task.is_expired():
        task.status = "expired"
        db.session.commit()
        abort(410, description="task expired")

    # running に進めるのは初回アクセス時のみ（冪等OK）
    if task.status == "pending":
        task.status = "running"
        db.session.commit()

    return jsonify({
        "provider": task.provider,
        "payload": task.payload or {},
        "status": task.status,
        "expires_at": task.expires_at.isoformat() + "Z",
        "message": task.message,
    })

@bp.get("/<token>/verification-link")
def get_verification_link(token: str):
    """
    サーバ側が取得した検証URLをヘルパーへ渡す。
    無ければ 202 Accepted。
    """
    task = ExternalSignupTask.query.filter_by(token=token).first()
    if not task:
        abort(404, description="task not found")
    if task.is_expired():
        task.status = "expired"
        db.session.commit()
        abort(410, description="task expired")

    if not task.verification_url:
        return jsonify({"status": "waiting"}), 202

    return jsonify({"status": "ready", "url": task.verification_url})

@bp.post("/<token>/done")
def complete_task(token: str):
    """
    ヘルパーが完了通知（ログイン不要：トークン認証）
    入力: {result:object, message?:string}
    動作: status=done にし結果を保存
    """
    task = ExternalSignupTask.query.filter_by(token=token).first()
    if not task:
        abort(404, description="task not found")

    body = _require_json()
    result = body.get("result") or {}
    message = body.get("message")

    # 期限切れでも結果を受け取り、状態を確定（監査のため）
    task.result = result
    task.message = message
    task.status = "done"
    task.updated_at = _now()
    db.session.commit()

    return jsonify({"ok": True})
