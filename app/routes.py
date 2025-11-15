from __future__ import annotations
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)
from flask import current_app  # 既存で使用、明示
from sqlalchemy.exc import OperationalError  # ✅ 追加：safe_commit 用

from flask import (
    Blueprint, render_template, redirect, url_for,
    flash, request, abort, g, jsonify, current_app, send_from_directory, session
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from pytz import timezone
from sqlalchemy import asc, nulls_last
from sqlalchemy.orm import selectinload

from app.extensions import func

from . import db
from .models import User, Article, PromptTemplate, Site, Keyword, Genre
from .forms import (
    LoginForm, RegisterForm,
    GenerateForm, PromptForm, ArticleForm, SiteForm, 
    ProfileForm
)
from .article_generator import enqueue_generation
from .wp_client import post_to_wp, _decorate_html, fetch_single_post

# --- 既存の import の下に追加 ---
import re
import os
import logging
import openai
import threading
import datetime
from .image_utils import fetch_featured_image  # ← ✅ 正しい
from collections import defaultdict
from urllib.parse import quote, urlsplit

from .article_generator import (
    _unique_title,
    _compose_body,
    _generate,
)
from app.forms import EditKeywordForm
from .forms import KeywordForm
from app.image_utils import _is_image_url

from app.services.blog_signup.livedoor_signup import generate_livedoor_id_candidates
from app.services.blog_signup.livedoor_atompub_recover import open_create_tab_for_handoff
# === Title & Meta バッチ再生成（管理API）で呼ぶ関数 ===
from app.tasks import run_title_meta_backfill

# ==== 外部SEO: 簡易ステータスストア（トークン→状態） ====
EXTSEO_STATUS = {}  # { token: { step, progress, captcha_url, site_id, account_id, ... } }

def _extseo_update(token: str, **kv):
    """外部SEOステータスをマージ更新（progressは0-100に丸める）"""
    st = dict(EXTSEO_STATUS.get(token) or {})
    for k, v in kv.items():
        if v is None:
            continue
        if k == "progress" and isinstance(v, (int, float)):
            v = max(0, min(100, int(v)))
        st[k] = v
    EXTSEO_STATUS[token] = st
    return st



JST = timezone("Asia/Tokyo")
bp = Blueprint("main", __name__)

# 必要なら app/__init__.py で admin_bp を登録
admin_bp = Blueprint("admin", __name__)

from app import db
from app.models import User, Site, Article
# リライト計画テーブル：存在名に合わせて import。なければ fallback で text() を使う
try:
    from app.models import ArticleRewritePlan
except Exception:
    ArticleRewritePlan = None
from sqlalchemy import text as _sql_text
from app.tasks import rewrite_enqueue_for_user
from app.tasks import _rewrite_retry_job, _serp_warmup_nightly_job
from concurrent.futures import ThreadPoolExecutor
_ui_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ui-triggers")

# --- Topic API: ヘッダトークン認証ヘルパ ---
def _topic_api_authorized() -> tuple[bool, int | None]:
    """
    X-Topic-Token を検証して (ok, user_id) を返す。
    - ログイン不要で叩くための軽量API鍵
    - 現状は環境変数 or 固定値 'local-test-token' を許可
    """
    try:
        token = (request.headers.get("X-Topic-Token") or "").strip()
        allowed = {t for t in (os.getenv("TOPIC_API_TOKEN"), "local-test-token") if t}
        if token and token in allowed:
            uid = int(os.getenv("TOPIC_API_USER_ID", "1"))
            return True, uid
    except Exception:
        current_app.logger.exception("[topic_api] token parse/verify failed")
    return False, None


# === Impersonation helpers =====================================================
# 置き場所：bp/admin_bp を作った直後（最初のルート定義より前）

def is_admin_effective() -> bool:
    """
    現在ログイン中のユーザーが管理者、または admin_id をセッションに保持している
    （=管理者がなりすまし中）なら True
    """
    try:
        return (
            getattr(current_user, "is_authenticated", False)
            and (getattr(current_user, "is_admin", False) or session.get("admin_id"))
        )
    except Exception:
        return False

from functools import wraps

def admin_required_effective(view_func):
    """
    なりすまし中でも管理者権限を維持している場合は通すデコレーター
    """
    @wraps(view_func)
    @login_required
    def _wrapped(*args, **kwargs):
        if not is_admin_effective():
            abort(403)
        return view_func(*args, **kwargs)
    return _wrapped

@admin_bp.route("/admin/return")
@login_required
def admin_return():
    """
    セッションに保存した admin_id に戻る（管理者へ復帰）
    """
    admin_id = session.get("admin_id")
    if not admin_id:
        flash("管理者セッションが見つかりません。", "warning")
        return redirect(url_for("main.dashboard", username=current_user.username))

    admin = User.query.get(admin_id)
    if not admin:
        session.pop("admin_id", None)
        flash("管理者アカウントが存在しません。", "danger")
        return redirect(url_for("main.login"))

    login_user(admin)
    session.pop("admin_id", None)
    flash("管理者に戻りました。", "info")
    return redirect(url_for("admin.admin_users"))
# ==============================================================================

# ------------------------------------------------------------------------------
# 管理API: Title & Meta バッチ再生成
#
# ・タイトル：記事タイトルをそのまま <title> として利用（DB更新は不要）
# ・メタ説明：AIで自動生成（既定180文字）。既存記事へ一括適用。
#
# 
# 受け取るパラメータ（GET/POSTとも可）:
#   - site_id: int       … 対象サイト限定（省略可）
#   - user_id: int       … 対象ユーザー限定（省略可）
#   - limit: int         … 1回の処理上限（既定 200）
#   - dryrun: bool       … プレビューのみ（DB書込なし）。true/1/on で有効
#   - push_to_wp: bool   … DB反映後に WP へも同期（posted 記事のみ）。dryrun時は無視
#   - after_id: int      … 続き実行用カーソル（前回レスポンスの cursor を渡す）
#
# 例:
#   GET  /admin/tools/title-meta-backfill?site_id=1&limit=200&dryrun=1
#   POST /admin/tools/title-meta-backfill  （JSON/FORM で同パラメータ）
# ------------------------------------------------------------------------------
def _as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")

def _as_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default
    
import time
import os
from flask import render_template, request, jsonify, current_app
try:
    from flask_wtf.csrf import csrf_exempt
except Exception:
    csrf_exempt = lambda f: f  # WTForms未使用環境でも動かすためのフォールバック
from app import db
from sqlalchemy.orm import load_only
from sqlalchemy import func, case, or_

try:
    # あなたのプロジェクトの User / Site / Article モデル名に合わせて import
    from app.models import User, Site, Article
except Exception:
    User = None
    Site = None
    Article = None

@admin_bp.route("/admin/tools/title-meta-backfill", methods=["GET", "POST"])
@admin_required_effective
@csrf_exempt  # ← このAPIだけCSRF免除（確実に通す）
def admin_title_meta_backfill():
    """
    Title & Meta バッチ再生成（UI簡略版）
    - GET: 超軽量描画（DBアクセスなし）でテンプレへ
    - POST: user_id のみ受け取り、バッチ処理を自動で最後まで実行（DB反映 + posted は WP 同期）
    """
    # ---- GET: 超軽量描画（DBヒット禁止）----
    if request.method == "GET":
        t0 = time.perf_counter()
        # このページは初回表示を最速にするため、ユーザー/サイトのDB取得を行わない
        # ユーザー候補はテンプレ側で /admin/tools/_users をAJAX遅延取得する前提
        users, sites = [], []
        dt = int((time.perf_counter() - t0) * 1000)
        current_app.logger.info("[admin:title-meta] FAST render (no DB) in %dms", dt)
        return render_template("admin/title_meta_backfill.html", users=users, sites=[])

    # ---- POST: “このユーザーの全記事に適用（自動で最後まで）” ----
    payload = (request.get_json(silent=True) or {}) or request.form.to_dict()
    current_app.logger.info("[admin:title-meta:POST] payload=%s", payload)
    user_id = _as_int(payload.get("user_id"))
    if not user_id:
        return jsonify({"ok": False, "error": "user_id is required"}), 400

    # 既定挙動：DB反映 + postedのみ WP 反映。limit は内部で十分大きくして周回数を減らす
    LIMIT_PER_CHUNK = _as_int(os.getenv("ADMIN_TM_LIMIT_PER_CHUNK", 500), 500) or 500
    MAX_ITERS       = _as_int(os.getenv("ADMIN_TM_MAX_ITERS", 200), 200) or 200

    from app.tasks import run_title_meta_backfill as _run_title_meta_backfill

    total_updated = 0
    iters = 0
    cursor = None
    last_result = {}
    # ★ 追加: WP反映の実績カウンタ（分子/分母/内訳）を合算
    wp_target_total_sum = 0
    wp_synced_ok_sum    = 0
    wp_unresolved_sum   = 0
    wp_failed_sum       = 0

    while True:
        iters += 1
        if iters > MAX_ITERS:
            current_app.logger.warning("[admin:title-meta] reached MAX_ITERS user_id=%s cursor=%s", user_id, cursor)
            break
        result = _run_title_meta_backfill(
            site_id=None,
            user_id=user_id,
            limit=LIMIT_PER_CHUNK,
            dryrun=False,
            after_id=cursor,
            push_to_wp=True,   # 旧「本適用 + WP反映」に相当
        )
        last_result = result
        if not result or not result.get("ok"):
            # 失敗は即終了
            status = 400
            err = (result or {}).get("error", "unknown error")
            return jsonify({"ok": False, "error": err, "updated": total_updated, "iterations": iters-1}), status

        # 1チャンクの更新件数（存在すれば）を加算
        total_updated += int(result.get("updated", 0))

        # ★ 追加: チャンクごとのWP実績を合算（キーが無い旧版でも0扱い）
        wp_target_total_sum += int(result.get("wp_target_total", 0) or 0)
        wp_synced_ok_sum    += int(result.get("wp_synced_ok", 0) or 0)
        wp_unresolved_sum   += int(result.get("wp_unresolved", 0) or 0)
        wp_failed_sum       += int(result.get("wp_failed", 0) or 0)

        # 続きカーソルのキー名は実装差異に合わせて両対応
        cursor = result.get("cursor") or result.get("next_after_id")
        done   = bool(result.get("done")) or (cursor in (None, "", 0))
        if done:
            break

    summary = {
        "ok": True,
        "user_id": user_id,
        "updated_total": total_updated,
        "iterations": iters,
        "last_cursor": cursor,
        "last_chunk": {
            "updated": int(last_result.get("updated", 0)),
            "cursor": last_result.get("cursor"),
            "done": bool(last_result.get("done")),
            # 参考: 最終チャンク単体のWP実績（UIで“直近の動き”を見たい場合に使用可）
            "wp_target_total": int(last_result.get("wp_target_total", 0) or 0),
            "wp_synced_ok":    int(last_result.get("wp_synced_ok", 0) or 0),
            "wp_unresolved":   int(last_result.get("wp_unresolved", 0) or 0),
            "wp_failed":       int(last_result.get("wp_failed", 0) or 0),
        },
        # ★ 合算（UIの分子/分母はこちらを利用）
        "wp_target_total": wp_target_total_sum,   # 分母: WP反映対象（postedのみ）
        "wp_synced_ok":    wp_synced_ok_sum,      # 分子: 実際にWPへ反映成功
        "wp_unresolved":   wp_unresolved_sum,     # 未解決（wp_post_id見つからず等）
        "wp_failed":       wp_failed_sum,         # API等の失敗
    }
    return jsonify(summary), 200


# --------------------------------------------------------------------
# ユーザー行（一覧テーブル）の軽量API
#   - 初回レンダは空HTML → このAPIでデータを遅延取得
#   - 検索: ?q=（username/email の部分一致）
#   - ページング: ?page=1&per_page=20
#   - 進捗メトリクス（本パッチで刷新）:
#       分母: 「投稿記事」数（既定: published_only=1）
#             → posted_at IS NOT NULL または posted_url <> ''
#       分子: meta_description が非空（COALESCE(...,'') <> ''）
# --------------------------------------------------------------------
@admin_bp.route("/admin/tools/title-meta-rows", methods=["GET"])
@admin_required_effective
def admin_title_meta_rows():
    if User is None or Site is None or Article is None:
        return jsonify({"items": [], "total": 0, "page": 1, "per_page": 20})

    q = (request.args.get("q") or "").strip()
    # 既定は「公開記事のみ」を分母にする（UIで総数ベースに変えたい時は ?published_only=0）
    published_only = (request.args.get("published_only", "1").strip().lower() in ("1", "true", "yes", "on"))
    try:
        page = max(1, int(request.args.get("page", "1")))
        per_page = max(1, min(50, int(request.args.get("per_page", "20"))))
    except Exception:
        page, per_page = 1, 20
    offset = (page - 1) * per_page

    # 対象ユーザー集合（検索・ページング）
    uq = db.session.query(User.id, User.username, User.email).order_by(User.id.asc())
    if q:
        like = f"%{q}%"
        uq = uq.filter(
            func.lower(User.username).like(func.lower(like)) |
            func.lower(User.email).like(func.lower(like))
        )
    total_users = uq.count()
    users = uq.offset(offset).limit(per_page).all()
    user_ids = [int(u.id) for u in users]
    if not user_ids:
        return jsonify({"items": [], "total": total_users, "page": page, "per_page": per_page})

    # 記事の下地（分母/分子ともこの集合から算出）
    # ← ここを()で括ってチェーンを改行。手動メタは分母から除外
    base_q = (
        db.session.query(
            Article.user_id.label("user_id"),
            Article.site_id.label("site_id"),
            func.coalesce(Article.meta_description, "").label("meta_description"),
            Article.posted_at.label("posted_at"),
            func.coalesce(Article.posted_url, "").label("posted_url"),
        )
        .filter(Article.user_id.in_(user_ids))
        .filter(Article.is_manual_meta == False)  # 分母から手動メタを外す
    )

    if published_only:
        base_q = base_q.filter(
            or_(
                Article.posted_at.isnot(None),
                func.coalesce(Article.posted_url, "") != ""
            )
        )

    base_sub = base_q.subquery()
    applied_cond = (func.coalesce(base_sub.c.meta_description, "") != "")

    # --- ユーザー別 集計（分母/分子/率） ---
    u_rows = (
        db.session.query(
            base_sub.c.user_id.label("user_id"),
            func.count(base_sub.c.user_id).label("total_cnt"),
            func.sum(case((applied_cond, 1), else_=0)).label("applied_cnt"),
        )
        .group_by(base_sub.c.user_id)
        .all()
    )
    totals_map   = {int(r.user_id): int(r.total_cnt or 0)   for r in u_rows}
    applied_map  = {int(r.user_id): int(r.applied_cnt or 0) for r in u_rows}

    # --- サイト別 内訳（ユーザーまとめて取得） ---
    s_rows = (
        db.session.query(
            base_sub.c.user_id.label("user_id"),
            base_sub.c.site_id.label("site_id"),
            func.count(base_sub.c.site_id).label("total"),
            func.sum(case((applied_cond, 1), else_=0)).label("applied"),
        )
        .group_by(base_sub.c.user_id, base_sub.c.site_id)
        .all()
    )

    # サイト名をまとめて引く
    site_ids = sorted({int(r.site_id) for r in s_rows if r.site_id is not None})
    site_map = {}
    if site_ids:
        for s in db.session.query(Site.id, Site.name, Site.url).filter(Site.id.in_(site_ids)).all():
            site_map[int(s.id)] = (s.name or s.url or f"site#{int(s.id)}")

    per_user_sites = {}
    for r in s_rows:
        uid = int(r.user_id)
        sid = int(r.site_id) if r.site_id is not None else 0
        if sid == 0:
            # site_id 無しはスキップ（集計としては残す場合はここを外す）
            continue
        per_user_sites.setdefault(uid, []).append({
            "site_id": sid,
            "name": site_map.get(sid, f"site#{sid}"),
            "total": int(r.total or 0),
            "applied": int(r.applied or 0),
            "percentage": float(round((float(r.applied or 0) / float(r.total)) * 100.0, 2)) if r.total else 0.0,
        })

    # レスポンス整形：整備対象が0でもユーザー行を返す
    items = []
    for u in users:
        uid = int(u.id)
        total = int(totals_map.get(uid, 0))
        applied = int(applied_map.get(uid, 0))
        target = max(total - applied, 0)  # 互換: 旧UI用フィールド
        pct = float(round((applied / total) * 100.0, 2)) if total else 0.0
        sites = sorted(per_user_sites.get(uid, []), key=lambda x: -x["total"])[:6]
        items.append({
            "user_id": uid,
            "username": u.username,
            "email": u.email,
            "total_cnt": total,
            "applied_cnt": applied,
            "percentage": pct,
            "target_cnt": target,   # ★ 互換のため残す（テンプレ更新後は不要）
            "sites": sites,
        })

    return jsonify({"items": items, "total": total_users, "page": page, "per_page": per_page})


# ------------------------------------------------------------------------------
# 一覧テーブル用の軽量API（1ユーザー=1行）
#   - 既存ページの初期描画はDBアクセスなしを維持。フロントが本APIをAJAX呼び出し
#   - 集計条件:
#       is_manual_meta = false
#       status IN ('done','posted')
#       meta_desc_quality IN ('empty','too_short','too_long','duplicate')
#   - 返却: ユーザー1行 + サイト内訳（横チップ向け）
# ------------------------------------------------------------------------------
@admin_bp.route("/admin/tools/title-meta-users", methods=["GET"])
@admin_required_effective
def admin_title_meta_users():
    if Article is None or User is None or Site is None:
        return jsonify({"users": []})

    # クエリ・パラメータ（必要最小限のみ）
    qualities = request.args.get("qualities")
    if qualities:
        quality_targets = tuple([q.strip() for q in qualities.split(",") if q.strip()])
    else:
        quality_targets = ("empty", "too_short", "too_long", "duplicate")

    try:
        limit_users = int(request.args.get("limit", "0"))  # 0=制限なし
        limit_users = max(0, limit_users)
    except Exception:
        limit_users = 0

    # 記事側の基礎集計（user_id, site_id ごと）
    base = (
        db.session.query(
            Article.user_id.label("user_id"),
            Article.site_id.label("site_id"),
            func.count(Article.id).label("targets"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted_targets"),
            func.sum(case((Article.status == "done",   1), else_=0)).label("done_targets"),
        )
        .filter(Article.is_manual_meta == False)  # noqa: E712
        .filter(Article.status.in_(("done", "posted")))
        .filter(Article.meta_desc_quality.in_(quality_targets))
        .group_by(Article.user_id, Article.site_id)
    )

    rows = base.all()
    if not rows:
        return jsonify({"users": []})

    # user→site内訳 へ整形
    per_user = {}
    user_ids = set()
    site_ids = set()
    for r in rows:
        uid = int(r.user_id)
        sid = int(r.site_id) if r.site_id is not None else 0
        user_ids.add(uid)
        if sid:
            site_ids.add(sid)
        item = per_user.setdefault(uid, {"user_id": uid, "targets": 0, "sites": []})
        item["targets"] += int(r.targets or 0)
        if sid:
            item["sites"].append({
                "site_id": sid,
                "targets": int(r.targets or 0),
                "posted":  int(r.posted_targets or 0),
                "done":    int(r.done_targets or 0),
            })

    # ユーザー名を最小集合だけ取得
    users_meta = {}
    if user_ids:
        q_users = (
            db.session.query(User.id, User.username)
            .filter(User.id.in_(list(user_ids)))
            .order_by(User.id.asc())
        )
        for u in q_users.all():
            users_meta[int(u.id)] = {"name": u.username}

    # サイト表示名を最小集合だけ取得
    sites_meta = {}
    if site_ids:
        q_sites = (
            db.session.query(Site.id, Site.name, Site.url)
            .filter(Site.id.in_(list(site_ids)))
            .order_by(Site.id.asc())
        )
        for s in q_sites.all():
            sites_meta[int(s.id)] = {"name": (s.name or s.url or f"site#{s.id}")}
    # 表示用に整形（サイトは名前を付与し、targets降順で並べる）
    users = []
    for uid, info in per_user.items():
        sites = info["sites"]
        # サイト名付与
        for s in sites:
            meta = sites_meta.get(int(s["site_id"]), {})
            s["name"] = meta.get("name", f"site#{s['site_id']}")
        # 降順
        sites.sort(key=lambda x: x["targets"], reverse=True)
        users.append({
            "user_id": uid,
            "name": users_meta.get(uid, {}).get("name", f"user#{uid}"),
            "targets": info["targets"],
            "sites": sites,
        })

    # ユーザーも targets 降順で並べる
    users.sort(key=lambda x: x["targets"], reverse=True)
    if limit_users and len(users) > limit_users:
        users = users[:limit_users]

    current_app.logger.info("[admin:title-meta:list] users=%s (qualities=%s)", len(users), ",".join(quality_targets))
    return jsonify({"users": users})



# ------------------------------------------------------------------------------
# 軽量サジェストAPI: ユーザー / サイト
# ------------------------------------------------------------------------------
@admin_bp.route("/admin/tools/_users", methods=["GET"])
@admin_required_effective
def admin_tools_users_suggest():
    """
    ?q= （username or email の部分一致）, ?limit=（既定20）
    """
    if User is None:
        return jsonify({"items": []})
    q = (request.args.get("q") or "").strip()
    try:
        limit = max(1, min(50, int(request.args.get("limit", "20"))))
    except Exception:
        limit = 20
    qry = db.session.query(User.id, User.username, User.email).order_by(User.id.asc())
    if q:
        like = f"%{q}%"
        qry = qry.filter(
            func.lower(User.username).like(func.lower(like)) |
            func.lower(User.email).like(func.lower(like))
        )
    rows = qry.limit(limit).all()
    items = [{"id": r.id, "label": f"#{r.id} {r.username} <{r.email}>" } for r in rows]
    return jsonify({"items": items})

@admin_bp.route("/admin/tools/_sites", methods=["GET"])
@admin_required_effective
def admin_tools_sites_suggest():
    """
    ?q= 部分一致, ?user_id= で絞込, ?limit=（既定20）
    """
    if Site is None:
        return jsonify({"items": []})
    q = (request.args.get("q") or "").strip()
    user_id = request.args.get("user_id")
    try:
        limit = max(1, min(50, int(request.args.get("limit", "20"))))
    except Exception:
        limit = 20
    qry = db.session.query(Site.id, Site.name, Site.url).order_by(Site.id.asc())
    if user_id:
        try:
            uid = int(user_id)
            qry = qry.filter(Site.user_id == uid)
        except Exception:
            pass
    if q:
        like = f"%{q}%"
        qry = qry.filter(
            func.lower(func.coalesce(Site.name, "")).like(func.lower(like)) |
            func.lower(Site.url).like(func.lower(like))
        )
    rows = qry.limit(limit).all()
    items = [{"id": r.id, "label": f"#{r.id} {r.name or r.url}"} for r in rows]
    return jsonify({"items": items})

# ------------------------------------------------------------------------------
# 進捗API: ユーザー別 Title/Meta 適用状況
#   分母: 全記事（status 不問）/ 公開記事のみは ?published_only=1
#   分子: meta_description が非空（+ 任意で meta_desc_last_updated_at IS NOT NULL）
# ------------------------------------------------------------------------------
@admin_bp.route("/admin/tools/title-meta-progress", methods=["GET"])
@admin_required_effective
def admin_title_meta_progress():
    if Article is None:
        return jsonify({"ok": False, "error": "Article model not available"}), 500
    try:
        user_id = int(request.args.get("user_id", "0"))
    except Exception:
        return jsonify({"ok": False, "error": "user_id is required"}), 400
    if user_id <= 0:
        return jsonify({"ok": False, "error": "user_id is required"}), 400

    published_only = (request.args.get("published_only", "0") in ("1", "true", "yes", "on"))

    base = db.session.query(Article).filter(Article.user_id == user_id)
    if published_only:
        # 公開済み判定: posted_at または posted_url のどちらかが入っていれば公開とみなす
        base = base.filter(or_(Article.posted_at.isnot(None), func.coalesce(Article.posted_url, "") != ""))

    # 以降でサブクエリ列だけを参照できるよう、必要列に絞って別名付け
    base_sub = (
        base.with_entities(
            Article.id.label("id"),
            Article.site_id.label("site_id"),
            Article.meta_description.label("meta_description"),
        ).subquery()
    )
    # サブクエリ列版の適用条件
    applied_cond_sub = func.coalesce(base_sub.c.meta_description, "") != ""

    # 分母：全件数
    total = db.session.query(func.count(base_sub.c.id)).scalar() or 0
    # 分子：meta_description が非空
    applied = (
        db.session.query(
            func.sum(case((applied_cond_sub, 1), else_=0))
        ).scalar() or 0
    )

    by_site_rows = (
        db.session.query(
            base_sub.c.site_id.label("site_id"),
            func.count(base_sub.c.id).label("total"),
            func.sum(case((applied_cond_sub, 1), else_=0)).label("applied"),
        )
        .select_from(base_sub)
        .group_by(base_sub.c.site_id)
        .order_by(base_sub.c.site_id)
        .limit(500)
        .all()
    )
    by_site = [{"site_id": int(r.site_id), "total": int(r.total), "applied": int(r.applied or 0)} for r in by_site_rows]

    pct = (applied / total * 100.0) if total else 0.0
    return jsonify({
        "ok": True,
        "user_id": user_id,
        "published_only": published_only,
        "total": total,
        "applied": applied,
        "percentage": round(pct, 2),
        "by_site": by_site,
    })


@bp.route('/robots.txt')
def robots_txt():
    return send_from_directory('static', 'robots.txt')

# routes.py または api.py 内

from app.models import User, ChatLog, GSCConfig
from datetime import datetime

@bp.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    username = data.get("username", "ユーザー")

    if not user_msg:
        return jsonify({"reply": "メッセージが空です。"})

    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"reply": "ユーザーが見つかりません。"})

        # 過去の履歴（最新10件）
        logs = ChatLog.query.filter_by(user_id=user.id).order_by(ChatLog.timestamp.desc()).limit(10).all()
        logs = list(reversed(logs))  # 時系列順にする

        # 会話履歴を構成
        messages = [
            {
                "role": "system",
                "content": f"あなたはVER12.AI-posting-tool『site craft』専属のAIアシスタントです。ユーザー（{username}さん）を名前で呼びながら、親しみやすくサポートしてください。"
            }
        ]
        for log in logs:
            messages.append({"role": log.role, "content": log.content})

        messages.append({"role": "user", "content": user_msg})

        # OpenAI呼び出し
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        # ✅ DBに保存
        db.session.add(ChatLog(user_id=user.id, role="user", content=user_msg))
        db.session.add(ChatLog(user_id=user.id, role="assistant", content=reply))
        db.session.commit()

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"エラー：{str(e)}"})


import stripe
from app import db
from app.models import User, UserSiteQuota, PaymentLog

stripe_webhook_bp = Blueprint('stripe_webhook', __name__)

@stripe_webhook_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = current_app.config["STRIPE_WEBHOOK_SECRET"]

    # ログ出力：受信記録
    current_app.logger.info("📩 Stripe Webhook Received")
    current_app.logger.info(payload.decode("utf-8"))

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        current_app.logger.error("❌ Webhook signature verification failed")
        return "Webhook signature verification failed", 400
    except Exception as e:
        current_app.logger.error(f"❌ Error parsing webhook: {str(e)}")
        return f"Error parsing webhook: {str(e)}", 400

    # PaymentIntent（通常購入も特別購入もここで処理）
    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        metadata = intent.get("metadata", {})

        user_id = metadata.get("user_id")
        site_count = int(metadata.get("site_count", 1))
        plan_type = metadata.get("plan_type", "affiliate")
        special = metadata.get("special", "no")
        stripe_payment_id = intent.get("id")

        # 値のチェック
        if special not in ["yes", "no"]:
            current_app.logger.warning(f"⚠️ 無効な special フラグ：{special}")
            return jsonify({"message": "Invalid special flag"}), 400

        if not user_id:
            current_app.logger.warning("⚠️ metadata に user_id が含まれていません")
            return jsonify({"message": "Missing user_id"}), 400

        # SiteQuotaLogでの冪等性チェック（重複処理防止）
        existing_quota_log = SiteQuotaLog.query.filter_by(stripe_payment_id=stripe_payment_id).first()
        if existing_quota_log:
            current_app.logger.warning("⚠️ この支払いはすでにQuotaに反映済みです")
            return jsonify({"message": "Quota already granted"}), 200

        user = User.query.get(int(user_id))
        if not user:
            current_app.logger.warning(f"⚠️ user_id={user_id} のユーザーが見つかりません")
            return jsonify({"message": "User not found"}), 400

        # Quota加算処理
        quota = UserSiteQuota.query.filter_by(user_id=user.id).first()
        if not quota:
            quota = UserSiteQuota(user_id=user.id, total_quota=0, used_quota=0, plan_type=plan_type)
            db.session.add(quota)

        quota.total_quota += site_count
        quota.plan_type = plan_type
        db.session.commit()

        current_app.logger.info(
            f"✅ Quota加算: user_id={user.id}, plan={plan_type}, site_count={site_count}, special={special}"
        )

        # SiteQuotaLogに履歴を保存
        quota_log = SiteQuotaLog(
            user_id=user.id,
            stripe_payment_id=stripe_payment_id,
            site_count=site_count,
            reason="Stripe支払い"
        )
        db.session.add(quota_log)
        db.session.commit()

        # PaymentLog保存処理
        amount = intent.get("amount")   # ✅ 正確な小数で保持
        email = intent.get("receipt_email") or intent.get("customer_email")

        charge_id = intent.get("latest_charge")
        charge = stripe.Charge.retrieve(charge_id)
        balance_tx_id = charge.get("balance_transaction")
        balance_tx = stripe.BalanceTransaction.retrieve(balance_tx_id)

        if not email:
            email = user.email

        fee = balance_tx.fee # ✅ 小数で保持
        net = balance_tx.net

        log = PaymentLog(
            user_id=user.id,
            email=email,
            amount=amount,
            fee=fee,
            net_income=net,
            plan_type=plan_type,
            stripe_payment_id=stripe_payment_id,
            status="succeeded"
        )
        db.session.add(log)
        db.session.commit()

        current_app.logger.info(f"💰 PaymentLog 保存：{email} ¥{amount}")

    return jsonify(success=True)


# Stripe APIキーを読み込み
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ────────────── create-payment-intent
@bp.route("/create-payment-intent", methods=["POST"])
def create_payment_intent():
    try:
        data = request.get_json()

        # ✅ 値の取得とバリデーション
        user_id = data.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required")

        plan_type = data.get("plan_type", "affiliate")
        site_count = int(data.get("site_count", 1))
        special = data.get("special", "no")

        # ✅ special, plan_type のバリデーション
        if special not in ["yes", "no"]:
            raise ValueError(f"Invalid special value: {special}")
        if plan_type not in ["affiliate", "business"]:
            raise ValueError(f"Invalid plan_type: {plan_type}")

        user_id = int(user_id)  # ✅ int変換は後にする（エラー対処のため）

        # 🔸 特別プランかどうかで価格を設定
        if special == "yes":
            unit_price = 1000
        else:
            unit_price = 5000 if plan_type == "affiliate" else 20000

        total_amount = unit_price * site_count

        # ✅ Stripe PaymentIntent を作成
        intent = stripe.PaymentIntent.create(
            amount=total_amount,
            currency="jpy",
            automatic_payment_methods={"enabled": True},
            payment_method_options={
                "card": {
                    "request_three_d_secure": "any"
                }
            },
            metadata={  # ✅ Webhookで必要な情報をすべて埋め込む
                "user_id": str(user_id),
                "plan_type": plan_type,
                "site_count": str(site_count),
                "special": special
            }
        )

        # ✅ 成功ログ（デバッグしやすく）
        current_app.logger.info(
            f"✅ PaymentIntent 作成: user_id={user_id}, plan_type={plan_type}, site_count={site_count}, special={special}, amount={total_amount}"
        )

        return jsonify({"clientSecret": intent.client_secret})

    except Exception as e:
        import traceback
        current_app.logger.error(f"[create-payment-intent エラー] {e}")
        current_app.logger.error(traceback.format_exc())
        return jsonify(error=str(e)), 400

# ────────────── 通常購入ページ
@bp.route("/purchase", methods=["GET", "POST"])
@login_required
def purchase():
    if request.method == "POST":
        plan_type = request.form.get("plan_type")
        site_count = int(request.form.get("site_count", 1))

        if plan_type == "affiliate":
            price_id = os.getenv("STRIPE_PRICE_ID_AFFILIATE")
        elif plan_type == "business":
            price_id = os.getenv("STRIPE_PRICE_ID_BUSINESS")
        else:
            price_id = None

        if not price_id:
            flash("不正なプランが選択されました。", "error")
            return redirect(url_for("main.purchase"))

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=current_user.email,
            line_items=[{
                "price": price_id,
                "quantity": site_count,
            }],
            mode="payment" if plan_type == "affiliate" else "subscription",
            success_url=url_for("main.purchase", _external=True) + "?success=true",
            cancel_url=url_for("main.purchase", _external=True) + "?canceled=true",
            metadata={
                "user_id": current_user.id,
                "plan_type": plan_type,
                "site_count": site_count
            }
        )
        return redirect(session.url, code=303)

    return render_template("purchase.html")


# ────────────── 特別プランページ（テンプレート表示）
@bp.route("/<username>/special-purchase", methods=["GET"])
@login_required
def special_purchase(username):
    if current_user.username != username:
        abort(403)

    # 管理者 or 管理者モードなら常に許可 / それ以外は is_special_access 必須
    is_admin = bool(getattr(current_user, "is_admin", False) or session.get("admin_id"))
    if not is_admin and not getattr(current_user, "is_special_access", False):
        flash("このページにはアクセスできません。", "danger")
        return redirect(url_for("main.dashboard", username=username))

    return render_template(
        "special_purchase.html",
        stripe_public_key=os.getenv("STRIPE_PUBLIC_KEY"),
        username=username
    )


import traceback

@admin_bp.route("/admin/sync-stripe-payments", methods=["POST"])
@admin_required_effective
def sync_stripe_payments():

    try:
        response = stripe.PaymentIntent.list(limit=100)
        data = response.data
        print(f"🔍 取得した決済件数: {len(data)}")

        for pi in data:
            payment_id = pi.id
            amount = pi.amount
            created_at = datetime.datetime.fromtimestamp(pi.created).strftime("%Y-%m-%d %H:%M")
            charge_id = pi.latest_charge

            email = (
                pi.get("receipt_email")
                or pi.get("customer_email")
                or "不明"
            )

            print(f"🧾 {created_at} | ¥{amount} | {payment_id} | email: {email} | チャージID: {charge_id}")

        return jsonify({"message": f"{len(data)} 件の決済を取得しました。ログを確認してください。"})

    except Exception as e:
        print("❌ エラー:", e)
        traceback.print_exc()
        return jsonify({"error": "処理中にエラーが発生しました"}), 500



@admin_bp.route("/admin/update-fee", methods=["POST"])
@admin_required_effective
def update_manual_fee():
    try:
        data = request.get_json()
        log_id = data.get("log_id")
        fee = data.get("manual_fee")

        if log_id is None or fee is None:
            return jsonify({"error": "不正なリクエスト"}), 400

        log = PaymentLog.query.get(log_id)
        if not log:
            return jsonify({"error": "該当するログが見つかりません"}), 404

        fee_int = int(fee)
        log.manual_fee = fee_int
        log.net_income = log.amount - fee_int  # ✅ 純利益を更新

        db.session.commit()

        return jsonify({"message": "手数料を保存しました"})
    except Exception as e:
        print("❌ 手数料保存中にエラー:", e)
        return jsonify({"error": "サーバーエラー"}), 500



# ────────────── 管理者ダッシュボード（セクション） ──────────────

from app.models import Article, User, PromptTemplate, Site
from os.path import exists, getsize

@admin_bp.route("/admin")
@admin_required_effective
def admin_dashboard():
    if not current_user.is_admin:
        flash("このページにはアクセスできません。", "error")
        return redirect(url_for("main.dashboard", username=current_user.username))

    # ✅ 重い画像チェック処理を削除して即リダイレクト
    return redirect(url_for("admin.admin_users"))


@admin_bp.route("/admin/prompts")
@admin_required_effective
def admin_prompt_list():

    users = User.query.order_by(User.last_name, User.first_name).all()
    return render_template("admin/prompts.html", users=users)


@admin_bp.route("/admin/keywords")
@admin_required_effective
def admin_keyword_list():

    # 全ユーザー取得（first_name/last_name順で表示順が安定）
    users = User.query.order_by(User.last_name, User.first_name).all()
    return render_template("admin/keywords.html", users=users)


@admin_bp.route("/admin/gsc-status")
@admin_required_effective
def admin_gsc_status():

    from app.models import Site, Article, User, GSCConfig
    from sqlalchemy import case

    # 各サイトの投稿数・GSC設定を取得
    results = (
        db.session.query(
            Site.id,
            Site.name,
            Site.url,
            Site.plan_type,
            User.name.label("user_name"),
            func.count(Article.id).label("article_count"),
            func.max(GSCConfig.id).label("gsc_configured")
        )
        .join(User, Site.user_id == User.id)
        .outerjoin(Article, Article.site_id == Site.id)
        .outerjoin(GSCConfig, GSCConfig.site_id == Site.id)
        .group_by(Site.id, User.id)
        .order_by(func.count(Article.id).desc())
        .all()
    )

    return render_template("admin/gsc_status.html", results=results)


# --- ダッシュボード強化系ルート ---

# 📊 統計サマリ（既存）
@admin_bp.route('/admin/dashboard')
@admin_required_effective
def admin_summary():
    return render_template("admin/dashboard.html")

# 🔄 処理中ジョブ一覧
@admin_bp.route("/admin/job-status")
@admin_required_effective
def job_status():
    processing_articles = Article.query.filter_by(status="gen").order_by(Article.created_at.desc()).all()
    return render_template("admin/job_status.html", articles=processing_articles)

# ─────────────────────────────────────────────────────────
# リライト ダッシュボード（管理用）
# ─────────────────────────────────────────────────────────
@admin_bp.route("/admin/rewrite", methods=["GET"])
def admin_rewrite_dashboard():
    """
    管理UI（ユーザー行ごとの一覧）。データは JSON API で取得。
    """
    # 一覧はテンプレ＋フロント側のAJAXで描画
    return render_template("admin/rewrite.html")

# ─────────────────────────────────────────
# 全体サマリ API（統一定義：queued/running=plans[is_active=TRUE], success/error/unknown=logs[記事ごとの最新版]）
# ─────────────────────────────────────────
from sqlalchemy import text as _sql_text  # ← raw SQL 用
from sqlalchemy import func, case, text
@admin_bp.route("/admin/rewrite/summary", methods=["GET"])
def admin_rewrite_summary():
    from app import redis_client
    # 全期間・統一定義（期間フィルタなし）。唯一の真実源は vw_rewrite_state
    cache_key = "admin:rewrite:summary:v7:scope=all"
    cached = redis_client.get(cache_key)
    if cached:
        return jsonify(json.loads(cached))

    try:
        # 統一定義：vw_rewrite_state から全期間集計
        agg_sql = _sql_text("""
            SELECT
              COUNT(*)                                                     AS target_articles,
              SUM((final_bucket='waiting')::int)                           AS queued,
              SUM((final_bucket='running')::int)                           AS running,
              SUM((final_bucket='success')::int)                           AS success,
              SUM((final_bucket='failed')::int)                            AS failed,
              SUM((final_bucket NOT IN ('waiting','running','success','failed')
                   OR final_bucket IS NULL)::int)                          AS unknown,
              MAX(GREATEST(COALESCE(log_executed_at, 'epoch'::timestamptz),
                           COALESCE(plan_created_at,'epoch'::timestamptz))) AS last_activity_at
            FROM vw_rewrite_state
        """)
        row = dict(db.session.execute(agg_sql).mappings().first() or {})
        totals = {
            "target_articles": int(row.get("target_articles", 0) or 0),
            "queued":          int(row.get("queued", 0) or 0),
            "running":         int(row.get("running", 0) or 0),
            "success":         int(row.get("success", 0) or 0),
            # 既存フロント互換のためキー名は "error" を維持（failed を error に載せ替え）
            "error":           int(row.get("failed", 0) or 0),
        }
        unknown = int(row.get("unknown", 0) or 0)
        last_activity_at = row.get("last_activity_at").isoformat() if row.get("last_activity_at") else None
    except Exception as e:
        current_app.logger.warning("[rewrite_summary] fallback: %s", e)
        totals = {"queued": 0, "running": 0, "success": 0, "error": 0}
        unknown = 0
        last_activity_at = None

    # レスポンス整形（unknown を追加しても既存UIに影響なし／欲しければ利用可能）
    payload = {
        "totals": totals,
        "unknown": unknown,
        "last_activity_at": last_activity_at,
        "scope": "all",  # 全期間
        "version": 7
    }
    # TTL は短め（並行実行の揺れ吸収＋負荷軽減）
    redis_client.set(cache_key, json.dumps(payload, ensure_ascii=False), ex=20)
    return jsonify(payload)

# ─────────────────────────────────────────
# 共通集計ヘルパ：サイト単位の集計（全期間・統一定義）
# queued/running = plans(is_active=TRUE)
# success/error/unknown = logs(記事ごとの最新ログ)
# ─────────────────────────────────────────

def _rewrite_counts_for_site(user_id: int, site_id: int):
    agg_sql = _sql_text("""
        WITH latest_log AS (
          SELECT DISTINCT ON (site_id, article_id)
                 user_id, site_id, article_id, wp_status, executed_at
          FROM public.article_rewrite_logs
          ORDER BY site_id, article_id, executed_at DESC
        )
        SELECT
          COUNT(*) FILTER (WHERE p.is_active AND p.status = 'queued')  AS queued,
          COUNT(*) FILTER (WHERE p.is_active AND p.status = 'running') AS running,
          COUNT(*) FILTER (WHERE ll.wp_status = 'success')             AS success,
          COUNT(*) FILTER (WHERE ll.wp_status = 'error')               AS error,
          COUNT(*) FILTER (WHERE ll.wp_status = 'unknown')             AS unknown
        FROM public.article_rewrite_plans p
        LEFT JOIN latest_log ll
          ON ll.user_id    = p.user_id
         AND ll.site_id    = p.site_id
         AND ll.article_id = p.article_id
        WHERE p.user_id = :uid
          AND p.site_id = :sid
    """)
    row = db.session.execute(agg_sql, {"uid": user_id, "sid": site_id}).mappings().first() or {}
    return {
        "queued":  int(row.get("queued", 0) or 0),
        "running": int(row.get("running", 0) or 0),
        "success": int(row.get("success", 0) or 0),
        "error":   int(row.get("error", 0) or 0),
        "unknown": int(row.get("unknown", 0) or 0),
    }


# ─────────────────────────────────────────
# 共通集計ヘルパ：ユーザー別のサイト集計（全期間・統一定義）
# ─────────────────────────────────────────
def _rewrite_counts_for_user_sites(user_id: int):
    # 統一定義：vw_rewrite_state を唯一の真実源にする
    from sqlalchemy import text as _sql  # ← 既にモジュール上部で定義済みなら不要
    agg_sql = _sql("""
       SELECT
         v.site_id,
         COALESCE(s.name, '') AS site_name,
         COUNT(*)                                                  AS target_articles,
         SUM((v.final_bucket = 'waiting')::int)                    AS waiting,
         SUM((v.final_bucket = 'running')::int)                    AS running,
         SUM((v.final_bucket = 'success')::int)                    AS success,
         SUM((v.final_bucket = 'failed')::int)                     AS failed,
         MAX(GREATEST(COALESCE(v.log_executed_at, 'epoch'::timestamp),
                      COALESCE(v.plan_created_at, 'epoch'::timestamp))) AS last_update
       FROM vw_rewrite_state v
       LEFT JOIN public.site s
         ON s.id = v.site_id
       WHERE v.user_id = :uid
       GROUP BY v.site_id, s.name
       ORDER BY v.site_id
     """)
    rows = db.session.execute(agg_sql, {"uid": user_id}).mappings().all()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────
# 追加: ユーザー別サイト一覧（HTML）
# URL: /admin/rewrite/user/<user_id>
# ─────────────────────────────────────────
@admin_bp.route("/admin/rewrite/user/<int:user_id>", methods=["GET"])
def admin_rewrite_user_sites(user_id: int):
    # ユーザー情報をテンプレに渡す
    from app.models import User
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    # 全期間・統一定義での集計（テンプレ要件に合わせて rows を渡す）
    rows = _rewrite_counts_for_user_sites(user_id)
    return render_template(
        "admin/rewrite_user.html",
        user=user,
        rows=rows,
        back_url=url_for("admin.admin_rewrite_dashboard"),
    )


# ─────────────────────────────────────────
# 追加: サイト別のリライト記事一覧（HTML）
# URL: /admin/rewrite/user/<user_id>/site/<site_id>
# ─────────────────────────────────────────
@admin_bp.route("/admin/rewrite/user/<int:user_id>/site/<int:site_id>", methods=["GET"])
@login_required
def admin_rewrite_site_articles(user_id: int, site_id: int):
    if not current_user.is_admin:
        abort(403)
    """
    指定ユーザー×サイトの “最新状態” を一覧表示（統一ビュー基準）。
    ステータス絞り込み・簡易ページネーションに対応。
    """
    from sqlalchemy import text as _sql
    from app.models import User, Site, Article

    user = db.session.get(User, user_id)
    site = db.session.get(Site, site_id)
    if not user or not site or site.user_id != user_id:
        abort(404)

    # 全期間・統一定義でのヘッダ4指標＋unknown
    header_counts = _rewrite_counts_for_site(user_id, site_id)
    scope = "all"  # 全期間

    # クエリパラメータ
    status = (request.args.get("status") or "").strip().lower()
    page   = max(1, request.args.get("page", type=int) or 1)
    per    = min(100, max(10, request.args.get("per", type=int) or 50))

    # 許容ステータス（まずは success / failed の2系統に対応）
    allowed = {"success", "failed"}
    if status not in allowed:
        status = "success"

    # ── 統一ビューからサイトのサマリ（waiting/running/success/failed/other）
    from app.services.rewrite.state_view import fetch_site_totals
    totals = fetch_site_totals(user_id=user_id, site_id=site_id)
    stats = {
        "queued":  int(totals.get("waiting", 0)),
        "running": int(totals.get("running", 0)),
        "success": int(totals.get("success", 0)),
        "error":   int(totals.get("failed", 0)),
        "unknown": int(totals.get("other", 0)),
    }
    # 互換：テンプレが期待する display_error を常に数値で渡す
    stats["display_error"] = stats.get("error", 0)

    # ─────────────────────────────────────────
    # 一覧用IDを final_bucket で抽出（新しい順）
    # ─────────────────────────────────────────
    bucket = "success" if status == "success" else "failed"
    ids_sql = _sql("""
      SELECT article_id
      FROM vw_rewrite_state
      WHERE user_id = :uid AND site_id = :sid AND final_bucket = :bucket
      ORDER BY log_executed_at DESC NULLS LAST, plan_created_at DESC NULLS LAST, article_id DESC
      LIMIT :limit OFFSET :offset
    """)
    id_rows = db.session.execute(
        ids_sql,
        {"uid": user_id, "sid": site_id, "bucket": bucket, "limit": per, "offset": (page-1)*per}
    ).fetchall()
    article_ids = [int(r[0]) for r in id_rows]

    # 総件数（ページネーション用）
    total_sql = _sql("""
      SELECT COUNT(*) FROM vw_rewrite_state
       WHERE user_id = :uid AND site_id = :sid AND final_bucket = :bucket
    """)
    total_count = int(db.session.execute(
        total_sql, {"uid": user_id, "sid": site_id, "bucket": bucket}
    ).scalar() or 0)

    # 表示用の詳細（最新 success / 失敗系ログ）を取得
    rows = []
    if article_ids:
        if status == "success":
            # 最新 success ログ
            detail_sql = _sql("""
              WITH latest AS (
                SELECT
                  l.id         AS log_id,
                  l.article_id,
                  l.plan_id,
                  l.wp_post_id,
                  l.executed_at,
                  ROW_NUMBER() OVER (PARTITION BY l.article_id ORDER BY l.executed_at DESC, l.id DESC) AS rn
                FROM article_rewrite_logs l
                WHERE l.article_id = ANY(:ids) AND l.wp_status = 'success'
              )
              SELECT
                lt.log_id,
                a.id          AS article_id,
                a.title       AS title,
                lt.plan_id    AS plan_id,
                lt.wp_post_id AS wp_post_id,
                lt.executed_at AS executed_at
              FROM latest lt
              JOIN articles a ON a.id = lt.article_id
              WHERE lt.rn = 1
              ORDER BY lt.executed_at DESC NULLS LAST, a.id DESC
            """)
            rows = list(db.session.execute(detail_sql, {"ids": article_ids}).mappings())
        else:
            # 最新 failed 系ログ
            detail_sql = _sql("""
              WITH latest AS (
                SELECT
                  l.id         AS log_id,
                  l.article_id,
                  l.plan_id,
                  l.wp_post_id,
                  l.executed_at,
                  l.wp_status,
                  ROW_NUMBER() OVER (PARTITION BY l.article_id ORDER BY l.executed_at DESC, l.id DESC) AS rn
                FROM article_rewrite_logs l
                WHERE l.article_id = ANY(:ids)
                  AND l.wp_status IN ('failed','error','canceled','aborted','timeout','stale')
              )
              SELECT
                lt.log_id,
                a.id          AS article_id,
                a.title       AS title,
                lt.plan_id    AS plan_id,
                lt.wp_post_id AS wp_post_id,
                lt.executed_at AS executed_at,
                lt.wp_status  AS wp_status
              FROM latest lt
              JOIN articles a ON a.id = lt.article_id
              WHERE lt.rn = 1
              ORDER BY lt.executed_at DESC NULLS LAST, a.id DESC
            """)
            rows = list(db.session.execute(detail_sql, {"ids": article_ids}).mappings())

    # テンプレ互換：articles 配列を用意（id/title/status/updated_at/wp_url/posted_url…）
    articles = []
    _last_dt = None
    for r in rows:
        dt = r.get("executed_at")
        if dt and (_last_dt is None or dt > _last_dt):
            _last_dt = dt

        # 成功時のみWPリンク生成。失敗はリンク無し。
        if status == "success" and r.get("wp_post_id"):
            base = (getattr(site, "site_url", None) or getattr(site, "url", "") or "").rstrip("/")
            wp_url = f"{base}/?p={r.get('wp_post_id')}" if base else None
        else:
            wp_url = None

        articles.append({
            "id": r.get("article_id"),              # 一覧のID列は記事IDを表示
            "article_id": r.get("article_id"),
            "title": r.get("title"),
            "status": status,                       # ← 固定 'success' から実値へ
            "attempts": None,
            "updated_at": (dt.isoformat() if dt else None),
            "posted_url": None,
            "wp_url": wp_url,
            "plan_id": r.get("plan_id"),
            "log_id": r.get("log_id"),
        })
    last_updated = _last_dt.isoformat() if _last_dt else None

    # ページネーション情報を構築
    total_pages = (total_count + per - 1) // per if per > 0 else 1
    first_idx = ((page - 1) * per + 1) if total_count > 0 else 0
    last_idx  = min(page * per, total_count)
    prev_url = (url_for("admin.admin_rewrite_site_articles",
                        user_id=user_id, site_id=site_id,
                        status=status, page=page-1, per=per)
                if page > 1 else None)
    next_url = (url_for("admin.admin_rewrite_site_articles",
                        user_id=user_id, site_id=site_id,
                        status=status, page=page+1, per=per)
                if page * per < total_count else None)
    pagination = {
        "total": total_count,
        "page": page,
        "per": per,
        "pages": total_pages,
        "first": first_idx,
        "last": last_idx,
        "prev_url": prev_url,
        "next_url": next_url,
    }

    return render_template(
        "admin/rewrite_site_articles.html",
        user_id=user_id,
        site_id=site_id,
        site=site,
        articles=articles,
        header_counts=header_counts,
        scope=scope,
        stats=stats,
        last_updated=last_updated,
        status=status,      # ← 現在の表示ステータスをテンプレへ
        pagination=pagination,  # ← 追加
        per=per,            # ← 明示的に渡しておく（リンク引継ぎ用）
    )

# ─────────────────────────────────────────
# リライト詳細（修正方針 / ログ詳細）
# URL: /admin/rewrite/log/<log_id>
# ─────────────────────────────────────────
@admin_bp.route("/admin/rewrite/log/<int:log_id>", methods=["GET"])
@login_required
def admin_rewrite_log_detail(log_id: int):
    if not current_user.is_admin:
        abort(403)

    from app.models import ArticleRewriteLog, Article

    log = db.session.get(ArticleRewriteLog, log_id)
    if not log:
        abort(404)

    article = None
    if log.article_id:
        article = db.session.get(Article, log.article_id)

    # 関連とパンくず用の派生値を明示的に渡す
    plan = getattr(log, "plan", None)
    user_id = getattr(article, "user_id", None) if article else None
    site_id = getattr(article, "site_id", None) if article else None

    return render_template(
        "admin/rewrite_log_detail.html",
        log=log,
        article=article,
        plan=plan,
        user_id=user_id,
        site_id=site_id,
    )

@admin_bp.route("/admin/rewrite/users", methods=["GET"])
def admin_rewrite_users():
    """
    JSON: 管理UI用のユーザー一覧（サイト数 + リライト集計）
    定義：vw_rewrite_state を唯一の真実源とする。
    """
    from sqlalchemy import text as _sql
    from app import redis_client
    from app.models import User, Site
    import json

    q = (request.args.get("q", type=str) or "").strip()
    nocache = request.args.get("nocache", type=int) == 1
    cache_key = f"admin:rewrite:users:v8:q={q}"
    if not nocache:
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify({"ok": True, "items": json.loads(cached)})

    # --- 表示名生成 ---
    full_name_expr = func.trim(
        func.concat(
            func.coalesce(func.nullif(User.last_name, ""), ""),
            " ",
            func.coalesce(func.nullif(User.first_name, ""), ""),
        )
    )
    name_expr = func.coalesce(func.nullif(full_name_expr, ""), User.username, User.email)

    # --- 検索フィルタ ---
    filters = []
    if q:
        like = f"%{q}%"
        filters.append(name_expr.ilike(like) | User.username.ilike(like) | User.email.ilike(like))

    # --- vw_rewrite_state によるユーザー単位集計 ---
    agg_sql = _sql("""
        SELECT
          v.user_id AS uid,
          COUNT(*) AS target_articles,
          SUM((v.final_bucket='waiting')::int) AS queued,
          SUM((v.final_bucket='running')::int) AS running,
          SUM((v.final_bucket='success')::int) AS success,
          SUM((v.final_bucket='failed')::int)  AS error,
          MAX(GREATEST(
              COALESCE(v.log_executed_at, 'epoch'::timestamp),
              COALESCE(v.plan_created_at, 'epoch'::timestamp)
          )) AS last_activity_at
        FROM vw_rewrite_state v
        GROUP BY v.user_id
        ORDER BY v.user_id
    """)
    agg_rows = db.session.execute(agg_sql).mappings().all()
    agg_map = {r["uid"]: r for r in agg_rows}

    # --- サイト数 ---
    site_sq = (
        db.session.query(Site.user_id.label("uid"), func.count(Site.id).label("site_count"))
        .group_by(Site.user_id)
        .subquery()
    )

    # --- ユーザー情報を結合 ---
    rows = (
        db.session.query(
            User.id.label("user_id"),
            name_expr.label("name"),
            func.coalesce(site_sq.c.site_count, 0).label("site_count"),
        )
        .outerjoin(site_sq, site_sq.c.uid == User.id)
        .filter(*filters)
        .order_by(User.id.asc())
        .all()
    )

    # --- 結果整形 ---
    items = []
    for r in rows:
        uid = r.user_id
        a = agg_map.get(uid, {})
        items.append({
            "user_id": uid,
            "name": r.name,
            "site_count": int(r.site_count or 0),
            "queued": int(a.get("queued", 0) or 0),
            "running": int(a.get("running", 0) or 0),
            "success": int(a.get("success", 0) or 0),
            "error": int(a.get("error", 0) or 0),
            "last_activity_at": (
                a.get("last_activity_at").isoformat() if a.get("last_activity_at") else None
            ),
            "target_articles": int(a.get("target_articles", 0) or 0),
        })

    # --- キャッシュ ---
    if not nocache:
        try:
            redis_client.setex(cache_key, 5, json.dumps(items, ensure_ascii=False))
        except Exception:
            pass

    return jsonify({"ok": True, "items": items})


# ─────────────────────────────────────────
# 追加: 内部SEO風の一覧API（テンプレ互換のキー名で返却）
# ─────────────────────────────────────────
@admin_bp.route("/admin/rewrite/users_progress", methods=["GET"])
def admin_rewrite_users_progress():
    """
    JSON: 各ユーザーのサイト数とリライト進捗（queued/running/success/error/last_activity_at）
    返却キーは { ok, users: [...] } でテンプレと一致。
    実体は /admin/rewrite/users と同じ集計（5秒キャッシュ）。
    """
    from sqlalchemy import case
    from app import redis_client
    # ★ NameError対策
    from app.models import User, Site, ArticleRewritePlan

    q = (request.args.get("q", type=str) or "").strip()
    nocache = request.args.get("nocache", type=int) == 1
    cache_key = f"admin:rewrite:users_progress:v3:q={q}"
    if not nocache:
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify({"ok": True, "users": json.loads(cached)})

    # 表示名: (last_name + ' ' + first_name) -> username -> email
    full_name_expr = func.trim(
        func.concat(
            func.coalesce(func.nullif(User.last_name, ""), ""),
            " ",
            func.coalesce(func.nullif(User.first_name, ""), ""),
        )
    )
    name_expr = func.coalesce(func.nullif(full_name_expr, ""), User.username, User.email)

    site_sq = (
        db.session.query(Site.user_id.label("uid"), func.count(Site.id).label("site_count"))
        .group_by(Site.user_id)
        .subquery()
    )
    queued_cnt  = func.sum(case((ArticleRewritePlan.status == "queued", 1), else_=0))
    running_cnt = func.sum(case((ArticleRewritePlan.status.in_(["running","in_progress"]), 1), else_=0))
    last_act    = func.max(func.coalesce(ArticleRewritePlan.finished_at, ArticleRewritePlan.created_at))
    plan_sq = (
        db.session.query(
            ArticleRewritePlan.user_id.label("uid"),
            queued_cnt.label("queued"),
            running_cnt.label("running"),
            last_act.label("last_activity_at"),
        )
        .group_by(ArticleRewritePlan.user_id)
        .subquery()
    )

    # logs 側：ユーザー別 success/error
    from sqlalchemy import text as _sql
    logs_user_sql = _sql("""
      WITH latest AS (
        SELECT
          l.article_id,
          l.wp_status,
          a.user_id,
          ROW_NUMBER() OVER (PARTITION BY l.article_id ORDER BY l.executed_at DESC) AS rn
        FROM article_rewrite_logs l
        JOIN articles a ON a.id = l.article_id
      )
      SELECT
        user_id AS uid,
        SUM(CASE WHEN wp_status = 'success' THEN 1 ELSE 0 END)               AS success,
        SUM(CASE WHEN wp_status IN ('error','failed') THEN 1 ELSE 0 END)     AS error
      FROM latest
      WHERE rn = 1
      GROUP BY user_id
    """)
    logs_user_sq = db.session.execute(logs_user_sql).mappings().all()
    logs_user_map = { r["uid"]: {"success": int(r["success"] or 0), "error": int(r["error"] or 0)} for r in logs_user_sq }

    filters = []
    if q:
        like = f"%{q}%"
        filters.append(
            name_expr.ilike(like) | User.username.ilike(like) | User.email.ilike(like)
        )

    try:
        rows = (
            db.session.query(
                User.id.label("user_id"),
                name_expr.label("name"),
                func.coalesce(site_sq.c.site_count, 0).label("site_count"),
                func.coalesce(plan_sq.c.queued, 0).label("queued"),
                func.coalesce(plan_sq.c.running, 0).label("running"),
                plan_sq.c.last_activity_at.label("last_activity_at"),
            )
            .outerjoin(site_sq, site_sq.c.uid == User.id)
            .outerjoin(plan_sq, plan_sq.c.uid == User.id)
            .filter(*filters)
            .order_by(
                func.coalesce(plan_sq.c.queued, 0).desc(),
                func.coalesce(plan_sq.c.running, 0).desc(),
                plan_sq.c.last_activity_at.desc().nullslast(),
                User.id.asc(),
            )
            .all()
        )
    except Exception as e:
        current_app.logger.exception("[admin/rewrite/users_progress] query failed: %s", e)
        return jsonify({"ok": False, "users": [], "error": str(e)}), 500

    users = [{
        "user_id": r.user_id,
        "name": r.name,
        "site_count": int(r.site_count or 0),
        "queued": int(r.queued or 0),
        "running": int(r.running or 0),
        "success": int(logs_user_map.get(r.user_id, {}).get("success", 0)),
        "error":   int(logs_user_map.get(r.user_id, {}).get("error", 0)),
        "last_activity_at": (r.last_activity_at.isoformat() if r.last_activity_at else None),
    } for r in rows]

    if not nocache:
        try:
            redis_client.setex(cache_key, 2, json.dumps(users, ensure_ascii=False))
        except Exception:
            pass
    return jsonify({"ok": True, "users": users})


@admin_bp.route("/admin/rewrite/enqueue", methods=["POST"])
def admin_rewrite_enqueue():
    """
    JSON: 全記事リライトをユーザー単位で queued に投入。
    body: { user_id, site_ids?: [..], article_ids?: [..], priority?: number }
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        user_id = int(payload.get("user_id"))
        # "1,2,3" / [1,2] / "  " どれでも受ける
        def _to_int_list(v):
            if v is None or v == "":
                return None
            if isinstance(v, list):
                return [int(x) for x in v if str(x).strip().isdigit()]
            return [int(x) for x in str(v).replace("\n", ",").split(",") if x.strip().isdigit()]
        site_ids = _to_int_list(payload.get("site_ids"))
        article_ids = _to_int_list(payload.get("article_ids"))
        priority = float(payload.get("priority", 0.0))
        res = rewrite_enqueue_for_user(user_id, site_ids=site_ids, article_ids=article_ids, priority=priority)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        current_app.logger.exception("[admin/rewrite/enqueue] failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 400


@admin_bp.route("/admin/rewrite/progress", methods=["GET"])
def admin_rewrite_progress():
    """
    JSON: 進捗サマリを返す。
    query: user_id (optional)
    返却: { totals: {queued,running,success,error}, recent: [...], last_updated }
    """

    # ここで確実にモデルを読み込む（NameError対策）
    try:
        from app.models import ArticleRewritePlan, Article
    except Exception:
        ArticleRewritePlan = None
        Article = None

    uid = request.args.get("user_id", type=int)

    # ---- queued / running は plans（is_active=TRUE）からリアルタイム集計
    base_plans_q = db.session.query(ArticleRewritePlan).filter(ArticleRewritePlan.is_active.is_(True))
    if uid:
        base_plans_q = base_plans_q.filter(ArticleRewritePlan.user_id == uid)

    plan_agg = (
        db.session.query(
            func.sum(case((ArticleRewritePlan.status == "queued", 1), else_=0)).label("queued"),
            func.sum(case((ArticleRewritePlan.status.in_(["running","in_progress"]), 1), else_=0)).label("running"),
        )
        .filter(ArticleRewritePlan.is_active.is_(True))
        .filter(*( [ArticleRewritePlan.user_id == uid] if uid else [] ))
        .one()
    )

    # ---- success / error は“最新ログのみ”でカウント（uid があればユーザーの article に限定）
    where_by_user = "JOIN articles a ON a.id = l.article_id" + (" AND a.user_id = :uid" if uid else "")
    logs_sql = _sql_text(f"""
      WITH latest AS (
        SELECT
          l.article_id,
          l.wp_status,
          l.executed_at,
          ROW_NUMBER() OVER (PARTITION BY l.article_id ORDER BY l.executed_at DESC) AS rn
        FROM article_rewrite_logs l
        {where_by_user}
      )
      SELECT
        SUM(CASE WHEN wp_status = 'success' THEN 1 ELSE 0 END)                   AS success,
        SUM(CASE WHEN wp_status IN ('error','failed') THEN 1 ELSE 0 END)         AS error,
        MAX(executed_at)                                                          AS last_log_ts
      FROM latest
      WHERE rn = 1
    """)
    logs_row = db.session.execute(logs_sql, {"uid": uid} if uid else {}).mappings().first() or {}

    totals = {
        "queued":  int(getattr(plan_agg, "queued", 0) or 0),
        "running": int(getattr(plan_agg, "running", 0) or 0),
        "success": int(logs_row.get("success", 0) or 0),
        "error":   int(logs_row.get("error", 0) or 0),
    }

    # --- ユーザー別の実行フラグ（一覧ボタンの文言切替材料）
    users_snapshot = None
    user_snapshot  = None
    if uid:
        user_snapshot = {
            "user_id": uid,
            "queued":  totals["queued"],
            "running": totals["running"],
            "is_running": (totals["queued"] + totals["running"]) > 0,
        }
    else:
        # 一覧用：is_active=TRUE かつ queued/running の件数を user_id ごとに集計
        rows = (
            db.session.query(
                ArticleRewritePlan.user_id.label("user_id"),
                func.sum(case((ArticleRewritePlan.status == "queued", 1), else_=0)).label("queued"),
                func.sum(case((ArticleRewritePlan.status.in_(["running","in_progress"]), 1), else_=0)).label("running"),
            )
            .filter(ArticleRewritePlan.is_active.is_(True))
            .group_by(ArticleRewritePlan.user_id)
            .all()
        )
        users_snapshot = [
            {
                "user_id": r.user_id,
                "queued":  int(r.queued or 0),
                "running": int(r.running or 0),
                "is_running": (int(r.queued or 0) + int(r.running or 0)) > 0,
            }
            for r in rows
        ]

    # 最近30件は従来どおり plans を表示（UIのテーブル互換）
    try:
        recent_plans = (
            base_plans_q.order_by(
                func.coalesce(
                    ArticleRewritePlan.finished_at,
                    ArticleRewritePlan.started_at,
                    ArticleRewritePlan.scheduled_at,
                    ArticleRewritePlan.created_at
                ).desc(),
                ArticleRewritePlan.id.desc()
            ).limit(30).all()
        )
        # posted_url を紐付け
        a_ids = [p.article_id for p in recent_plans if getattr(p, "article_id", None)]
        art_map = {}
        if a_ids:
            arts = (db.session.query(Article.id, Article.posted_url)
                            .filter(Article.id.in_(a_ids)).all())
            art_map = {aid: url for (aid, url) in arts}
        recent = []
        for r in recent_plans:
            best_ts = r.finished_at or r.started_at or r.scheduled_at or r.created_at
            recent.append({
                "id": r.id,
                "article_id": r.article_id,
                "status": r.status,
                "attempts": getattr(r, "attempts", None),
                "updated_at": (best_ts.isoformat() if best_ts else None),
                "posted_url": art_map.get(r.article_id),
            })
    except Exception:
        # Fallback: テーブル名で素直に叩く
        where = "WHERE user_id=:uid" if uid else ""
        agg_rows = db.session.execute(
            _sql_text(f"SELECT status, COUNT(*) FROM article_rewrite_plans {where} GROUP BY status"),
            {"uid": uid} if uid else {},
        ).fetchall()
        totals = {r[0] or "": int(r[1] or 0) for r in agg_rows}
        totals["success"] = int(totals.get("success", 0)) + int(totals.get("done", 0))
        recent_rows = db.session.execute(
            _sql_text(f"""
              SELECT
                  id,
                  article_id,
                  status,
                  attempts,
                  COALESCE(finished_at, started_at, scheduled_at, created_at) AS updated_at
              FROM article_rewrite_plans
              {where}
              ORDER BY updated_at DESC NULLS LAST, id DESC
              LIMIT 30
            """),
            {"uid": uid} if uid else {},
        ).fetchall()
        a_ids = [row[1] for row in recent_rows if row[1]]
        art_map = {}
        if a_ids:
            arts = (db.session.query(Article.id, Article.posted_url)
                            .filter(Article.id.in_(a_ids)).all())
            art_map = {aid: url for (aid, url) in arts}
        recent = []
        for r in recent_rows:
            best_ts = r[4]
            recent.append({
                "id": r[0],
                "article_id": r[1],
                "status": r[2],
                "attempts": r[3],
                "updated_at": (best_ts.isoformat() if best_ts else None),
                "posted_url": art_map.get(r[1]),
            })
    return jsonify({
        "ok": True,
        "totals": {
            "queued": int(totals.get("queued", 0)),
            "running": int(totals.get("running", 0)),
            "success": int(totals.get("success", 0)),
            "error": int(totals.get("error", 0)),
        },
        # 追加：一覧で使うスナップショット（uid指定時は user_snapshot、未指定時は users_snapshot）
        "user": user_snapshot,
        "users": users_snapshot,
        "recent": recent,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    })


@admin_bp.route("/admin/rewrite/plans", methods=["GET"])
def admin_rewrite_plans():
    """
    JSON: 計画一覧をページング返却。
    query: user_id?<int>, status?<str>, page?<int>=1, per_page?<int>=50
    """
    uid = request.args.get("user_id", type=int)
    status = request.args.get("status", type=str)
    page = max(1, request.args.get("page", default=1, type=int))
    per_page = min(200, max(1, request.args.get("per_page", default=50, type=int)))

    where = []
    params = {}
    if uid:
        where.append("user_id=:uid")
        params["uid"] = uid
    if status:
        where.append("status=:st")
        params["st"] = status
    wsql = ("WHERE " + " AND ".join(where)) if where else ""

    rows = db.session.execute(
        _sql_text(f"""
          SELECT id, user_id, article_id, status, attempts, created_at, updated_at
            FROM article_rewrite_plans
           {wsql}
        ORDER BY updated_at DESC NULLS LAST, id DESC
           LIMIT :lim OFFSET :off
        """),
        {**params, "lim": per_page, "off": (page-1)*per_page},
    ).fetchall()
    data = [
        {
            "id": r[0], "user_id": r[1], "article_id": r[2], "status": r[3],
            "attempts": r[4],
            "created_at": (r[5].isoformat() if r[5] else None),
            "updated_at": (r[6].isoformat() if r[6] else None),
        }
        for r in rows
    ]
    return jsonify({"ok": True, "items": data, "page": page, "per_page": per_page})

# ─────────────────────────────────────────
# 管理API: retry_failed / serp_warmup
# ─────────────────────────────────────────
@admin_bp.route("/admin/rewrite/retry_failed", methods=["POST"])
def admin_rewrite_retry_failed():
    """
    失敗プランの再キューを即時トリガ。ノンブロッキングで実行。
    body: { user_id?:int, max_attempts?:int, min_age_minutes?:int, limit?:int }
    """
    payload = request.get_json(silent=True) or {}
    user_id      = payload.get("user_id")  # 受け取りのみ（ジョブが対応していれば利用）
    max_attempts = int(payload.get("max_attempts", 3))
    min_age_min  = int(payload.get("min_age_minutes", 30))
    limit        = int(payload.get("limit", 100))
    app_obj = current_app._get_current_object()
    def _run():
        try:
            # 内部はログに結果を出す（user_id対応の実装があれば渡す）
            try:
                _rewrite_retry_job(app_obj, user_id=user_id, max_attempts=max_attempts, min_age_minutes=min_age_min, limit=limit)
            except TypeError:
                # 旧シグネチャ互換
                _rewrite_retry_job(app_obj)
        except Exception as e:
            current_app.logger.exception("[admin/rewrite/retry_failed] job error: %s", e)
    _ui_executor.submit(_run)
    return jsonify({"ok": True, "queued": True, "params": {
        "user_id": user_id, "max_attempts": max_attempts, "min_age_minutes": min_age_min, "limit": limit
    }})

@admin_bp.route("/admin/rewrite/serp_warmup", methods=["POST"])
def admin_rewrite_serp_warmup():
    """
    SERP 温めを即時トリガ（夜間ジョブの手動発火相当）。ノンブロッキング。
    body: { user_id?:int, days?:int, limit?:int }
    """
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    days  = int(payload.get("days", 45))
    limit = int(payload.get("limit", 30))
    app_obj = current_app._get_current_object()
    def _run():
        try:
            # 夜間ジョブ本体を流用（user_id対応の実装があれば渡す）
            try:
                _serp_warmup_nightly_job(app_obj, user_id=user_id, days=days, limit=limit)
            except TypeError:
                _serp_warmup_nightly_job(app_obj)
        except Exception as e:
            current_app.logger.exception("[admin/rewrite/serp_warmup] job error: %s", e)
    _ui_executor.submit(_run)
    return jsonify({"ok": True, "queued": True, "params": {"user_id": user_id, "days": days, "limit": limit}})

import subprocess
from flask import jsonify

@admin_bp.route("/admin/log-stream")
@admin_required_effective
def log_stream():
    """最新の system.log を読み込んでJSONで返す（最大30行）"""
    try:
        from app.utils.log_utils import parse_logs
        log_path = os.path.join("logs", "system.log")

        # ログファイルの末尾から最大30行を取得
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-30:]

        # 1行ごとに整形
        logs = parse_logs(lines)
        return jsonify({"logs": logs})

    except Exception as e:
        import traceback
        print("❌ log_stream failed:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)})




# 🧠 API使用量／トークン分析
@admin_bp.route("/admin/api-usage")
@admin_required_effective
def api_usage():
    from app.models import TokenUsageLog, User
    from datetime import datetime
    # 日別集計（過去30日）
    today = datetime.utcnow().date()
    date_30_days_ago = today - timedelta(days=29)

    daily_data = (
        db.session.query(
            func.date(TokenUsageLog.created_at).label("date"),
            func.sum(TokenUsageLog.total_tokens).label("total_tokens")
        )
        .filter(TokenUsageLog.created_at >= date_30_days_ago)
        .group_by("date")
        .order_by("date")
        .all()
    )

    # ユーザー別集計（過去30日）
    user_data = (
        db.session.query(
            User.email,
            func.sum(TokenUsageLog.total_tokens).label("total_tokens")
        )
        .join(TokenUsageLog, TokenUsageLog.user_id == User.id)
        .filter(TokenUsageLog.created_at >= date_30_days_ago)
        .group_by(User.email)
        .order_by(func.sum(TokenUsageLog.total_tokens).desc())
        .all()
    )

    return render_template(
        "admin/api_usage.html",
        daily_data=daily_data,
        user_data=user_data
    )


# 💰 今月の売上＆取り分サマリ
@admin_bp.route("/admin/revenue-summary")
@admin_required_effective
def revenue_summary():
    from app.models import PaymentLog, User
    from datetime import datetime
    # 今月の開始日を取得（UTC）
    today = datetime.utcnow()
    first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # 今月の売上（成功した決済のみ）
    logs = (
        db.session.query(
            User.email,
            func.sum(PaymentLog.amount).label("total_amount"),
            func.count(PaymentLog.id).label("count")
        )
        .join(User, PaymentLog.user_id == User.id)
        .filter(PaymentLog.status == "succeeded")
        .filter(PaymentLog.created_at >= first_day)
        .group_by(User.email)
        .order_by(func.sum(PaymentLog.amount).desc())
        .all()
    )

    # 総売上
    total = sum(row.total_amount for row in logs)

    return render_template(
        "admin/revenue_summary.html",
        logs=logs,
        total=total
    )


# 📈 売上推移グラフ＋CSVダウンロード
# 📈 月別売上グラフ + CSVダウンロード
@admin_bp.route("/admin/revenue-graph")
@admin_required_effective
def revenue_graph():
    from app.models import PaymentLog
    from datetime import datetime, timedelta

    # 過去12ヶ月分の月次集計
    today = datetime.utcnow()
    first_day = today.replace(day=1) - timedelta(days=365)

    monthly_data = (
        db.session.query(
            func.to_char(PaymentLog.created_at, 'YYYY-MM').label("month"),
            func.sum(PaymentLog.amount).label("total")
        )
        .filter(PaymentLog.status == "succeeded")
        .filter(PaymentLog.created_at >= first_day)
        .group_by("month")
        .order_by("month")
        .all()
    )

    return render_template("admin/revenue_graph.html", monthly_data=monthly_data)

# 📥 CSVダウンロードルート
@admin_bp.route("/admin/download-revenue-log")
@admin_required_effective
def download_revenue_log():
    from app.models import PaymentLog, User
    import csv
    from io import StringIO
    from flask import Response

    logs = (
        db.session.query(
            PaymentLog.id,
            PaymentLog.amount,
            PaymentLog.status,
            PaymentLog.created_at,
            User.email
        )
        .join(User, PaymentLog.user_id == User.id)
        .order_by(PaymentLog.created_at.desc())
        .all()
    )

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Email", "金額（円）", "ステータス", "日時"])

    for log_id, amount, status, created_at, email in logs:
        writer.writerow([log_id, email, amount // 100, status, created_at.strftime("%Y-%m-%d %H:%M:%S")])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=revenue_log.csv"}
    )


# ─────────── 管理者：ジャンル管理（ユーザーごとのジャンル表示）
@admin_bp.route("/admin/genres", methods=["GET"])
@admin_required_effective
def manage_genres():
    if not current_user.is_admin:
        abort(403)

    from app.models import User  # 念のためUserをインポート
    users = User.query.order_by(User.last_name, User.first_name).all()

    return render_template("admin/genres.html", users=users)


@admin_bp.route("/admin/genres/delete/<int:genre_id>", methods=["POST"])
@admin_required_effective
def delete_genre(genre_id):

    genre = Genre.query.get_or_404(genre_id)
    db.session.delete(genre)
    db.session.commit()
    flash("ジャンルを削除しました", "info")
    return redirect(url_for("admin.manage_genres"))


@admin_bp.route("/admin/users", methods=["GET", "POST"])  # ✅ POST対応を追加
@admin_required_effective
def admin_users():

    # ✅ サイト枠追加リクエスト処理（POSTで来たときのみ）
    if request.method == "POST":
        if request.form.get("action") == "increase_quota":
            user_id = int(request.form.get("user_id"))
            plan_type = request.form.get("plan_type")

            # ✅ 該当ユーザー＆プランの枠を取得 or 作成
            quota = UserSiteQuota.query.filter_by(user_id=user_id, plan_type=plan_type).first()
            if quota:
                quota.total_quota += 1
            else:
                quota = UserSiteQuota(user_id=user_id, plan_type=plan_type, total_quota=1)
                db.session.add(quota)

            db.session.commit()
            flash("サイト枠を +1 しました", "success")

            return redirect(url_for("admin.admin_users"))

    # ✅ 必要最低限のユーザー情報のみ取得（→ Row形式 → dict形式に変換）
    raw_users = db.session.query(
        User.id,
        User.first_name,
        User.last_name,
        User.email,
        User.is_admin,
        User.is_special_access,
        User.created_at
    ).order_by(User.id).all()

    users = [
        {
            "id": u.id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "email": u.email,
            "is_admin": u.is_admin,
            "is_special_access": u.is_special_access,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "不明"
        }
        for u in raw_users
    ]

    site_count    = Site.query.count()
    prompt_count  = PromptTemplate.query.count()
    article_count = Article.query.count()

    # ✅ ここに追加
    stuck_counts = dict(
        db.session.query(
            Article.user_id,
            func.count()
        ).filter(
            Article.status.in_(["pending", "gen"])
        ).group_by(Article.user_id).all()
    )

    return render_template(
        "admin/users.html",
        users=users,  # ← JSONシリアライズ可能な形式に修正済み
        site_count=site_count,
        prompt_count=prompt_count,
        article_count=article_count,
        user_count=len(users),
        stuck_counts=stuck_counts
    )


@admin_bp.route("/api/admin/user_stats/<int:user_id>")
@admin_required_effective
def api_user_stats(user_id):

    from collections import defaultdict

    # 🔸 記事数
    total_articles = db.session.query(func.count(Article.id)).filter_by(user_id=user_id).scalar()

    # 🔸 途中記事（pending / gen）
    stuck_articles = db.session.query(func.count(Article.id)).filter(
        Article.user_id == user_id,
        Article.status.in_(["pending", "gen"])
    ).scalar()

    # 🔸 サイト枠（UserSiteQuota と Site 使用数の差）
    quota_rows = db.session.query(
        UserSiteQuota.plan_type,
        UserSiteQuota.total_quota
    ).filter_by(user_id=user_id).all()

    site_counts = db.session.query(
        Site.plan_type,
        func.count(Site.id)
    ).filter_by(user_id=user_id).group_by(Site.plan_type).all()

    # 整形：plan_type → { used, total, remaining }
    summary = {}
    used_map = {pt: c for pt, c in site_counts}

    for plan_type, total_quota in quota_rows:
        used = used_map.get(plan_type, 0)
        remaining = max(total_quota - used, 0)
        summary[plan_type] = {
            "used": used,
            "total": total_quota,
            "remaining": remaining
        }

    return jsonify({
        "article_count": total_articles,
        "stuck_count": stuck_articles,
        "quota_summary": summary
    })



@admin_bp.route("/admin/user/<int:uid>")
@admin_required_effective
def admin_user_detail(uid):

    user = User.query.get_or_404(uid)

    # 関連情報をすべて取得
    sites = Site.query.filter_by(user_id=uid).all()
    prompts = PromptTemplate.query.filter_by(user_id=uid).all()
    keywords = Keyword.query.filter_by(user_id=uid).all()
    articles = Article.query.filter_by(user_id=uid).order_by(Article.created_at.desc()).limit(20).all()
    payments = PaymentLog.query.filter_by(user_id=uid).order_by(PaymentLog.created_at.desc()).all()

    return render_template(
        "admin/user_detail.html",
        user=user,
        sites=sites,
        prompts=prompts,
        keywords=keywords,
        articles=articles,
        payments=payments
    )


from app.forms import QuotaUpdateForm

@admin_bp.route("/admin/quota-edit/<int:uid>", methods=["GET", "POST"])
@admin_required_effective
def admin_quota_edit(uid):

    user = User.query.get_or_404(uid)
    form = QuotaUpdateForm()

    if form.validate_on_submit():
        plan_type = form.plan_type.data
        count = form.count.data

        # クォータ取得 or 作成
        quota = UserSiteQuota.query.filter_by(user_id=user.id, plan_type=plan_type).first()
        if not quota:
            quota = UserSiteQuota(user_id=user.id, plan_type=plan_type, total_quota=0, used_quota=0)
            db.session.add(quota)

        quota.total_quota += count

        log = SiteQuotaLog(
            user_id=user.id,
            plan_type=plan_type,
            site_count=count,
            reason="管理者手動追加",
            created_at = datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        flash(f"✅ {plan_type}プランに{count}枠追加しました", "success")
        return redirect(url_for("admin.admin_users"))

    return render_template("admin/quota_edit.html", user=user, form=form)



@admin_bp.post("/admin/user/<int:uid>/toggle-special")
@admin_required_effective
def toggle_special_access(uid):
    # 管理者のみ許可

    # 対象ユーザー取得
    user = User.query.get_or_404(uid)

    # is_special_access をトグル（ON ⇔ OFF）
    user.is_special_access = not user.is_special_access
    db.session.commit()

    flash(f"{user.email} の特別アクセスを {'✅ 有効化' if user.is_special_access else '❌ 無効化'} しました。", "success")
    return redirect(url_for("admin.admin_users"))



@admin_bp.route("/admin/sites")
@admin_required_effective
def admin_sites():
    if not current_user.is_admin:
        flash("このページにはアクセスできません。", "error")
        return redirect(url_for("main.dashboard", username=current_user.username))

    from sqlalchemy import case, literal, func
    from app.models import Site, Article, User, Genre, GSCConfig, GSCDailyTotal
    from datetime import datetime, timezone, timedelta
    from collections import defaultdict

    # 🔹 ジャンルID→ジャンル名の辞書を事前取得
    genre_dict = {g.id: g.name for g in Genre.query.all()}

    # 🔹 GSCは「JSTの昨日まで」の直近28日で合計を出す（結合なし・相関サブクエリ）
    JST = timezone(timedelta(hours=9))
    _today_jst = datetime.now(timezone.utc).astimezone(JST).date()
    _end_d = _today_jst - timedelta(days=1)      # 昨日まで
    _start_d = _end_d - timedelta(days=27)       # 直近28日
    _gsc_clicks_28d = (
        db.session.query(func.coalesce(func.sum(GSCDailyTotal.clicks), 0))
        .filter(GSCDailyTotal.site_id == Site.id,
                GSCDailyTotal.date >= _start_d,
                GSCDailyTotal.date <= _end_d)
        .correlate(Site).scalar_subquery()
    )
    _gsc_impr_28d = (
        db.session.query(func.coalesce(func.sum(GSCDailyTotal.impressions), 0))
        .filter(GSCDailyTotal.site_id == Site.id,
                GSCDailyTotal.date >= _start_d,
                GSCDailyTotal.date <= _end_d)
        .correlate(Site).scalar_subquery()
    )

    # 🔹 サイトごとの統計情報（投稿数など）＋GSC接続状態を取得
    raw = (
        db.session.query(
            Site.id,
            Site.name,
            Site.url,
            Site.plan_type,
            Site.genre_id,
            Site.user_id,
            func.concat(User.last_name, literal(" "), User.first_name).label("user_name"),
            func.count(Article.id).label("total"),
            func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
            func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
            _gsc_clicks_28d.label("clicks"),
            _gsc_impr_28d.label("impressions"),
            func.max(GSCConfig.id).isnot(None).label("gsc_connected")
        )
        .join(User, Site.user_id == User.id)
        .outerjoin(Article, Site.id == Article.site_id)
        .outerjoin(GSCConfig, Site.id == GSCConfig.site_id)
        .group_by(Site.id, User.id)
        .order_by(User.id, Site.id.desc())
        .all()
    )

    # 🔹 ユーザー単位でまとめてテンプレートに渡すための構造を構築
    sites_by_user = defaultdict(lambda: {"user_id": None, "sites": [], "genres": set()})

    for row in raw:
        user_name = row.user_name
        genre_id = row.genre_id
        genre_name = genre_dict.get(genre_id, "") if genre_id else ""

        # 各サイトの情報
        site_info = {
            "id": row.id,  # ← ✅ この行を追加してください
            "name": row.name,
            "url": row.url,
            "plan_type": row.plan_type,
            "total": row.total or 0,
            "done": row.done or 0,
            "posted": row.posted or 0,
            "error": row.error or 0,
            "clicks": row.clicks or 0,
            "impressions": row.impressions or 0,
            "genre": genre_name,
            "gsc_connected": bool(row.gsc_connected)  # ← GSC接続ラベルに正しく対応
        }

        # 初回時のみ user_id を登録
        if sites_by_user[user_name]["user_id"] is None:
            sites_by_user[user_name]["user_id"] = row.user_id

        # サイト情報を格納
        sites_by_user[user_name]["sites"].append(site_info)

        # ジャンル名があれば追加（重複回避のため set）
        if genre_name:
            sites_by_user[user_name]["genres"].add(genre_name)

    # 🔹 最終的に genres をソートされたリストに変換（select要素用）
    for user_data in sites_by_user.values():
        user_data["genres"] = sorted(user_data["genres"])

    # 🔹 テンプレートに渡す
    return render_template("admin/sites.html", sites_by_user=sites_by_user)

@admin_bp.route('/admin/delete_site/<int:site_id>', methods=['POST'])
@admin_required_effective
def delete_site(site_id):

    site = Site.query.get_or_404(site_id)

    # ✅ 関連記事削除
    Article.query.filter_by(site_id=site.id).delete()

    # ✅ 関連キーワード削除
    Keyword.query.filter_by(site_id=site.id).delete()

    # ✅ GSC 認証トークン削除
    GSCAuthToken.query.filter_by(site_id=site.id).delete()

    # ✅ GSC 設定データ削除
    GSCConfig.query.filter_by(site_id=site.id).delete()

    # ❌ アイキャッチ画像ファイルは残す（/static/images/...）

    # ❌ StripeやTokenログ等は削除しない（監査用）

    # ✅ 最後にサイト本体を削除
    db.session.delete(site)
    db.session.commit()

    flash('サイトと関連データ（記事・キーワード・GSC情報）を削除しました。', 'success')
    return redirect(url_for('admin.admin_sites'))  # ✅ 修正済み



@admin_bp.route("/admin/user/<int:uid>/bulk-delete", methods=["POST"])
@admin_required_effective
def bulk_delete_articles(uid):

    # pending または gen 状態の記事を一括削除
    Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).delete()

    db.session.commit()
    flash("✅ 途中状態の記事を一括削除しました", "success")
    return redirect(url_for("admin.user_articles", uid=uid))



@admin_bp.post("/admin/delete-stuck-articles")
@admin_required_effective
def delete_stuck_articles():

    stuck = Article.query.filter(Article.status.in_(["pending", "gen"])).all()

    deleted_count = len(stuck)
    for a in stuck:
        db.session.delete(a)
    db.session.commit()

    flash(f"{deleted_count} 件の途中停止記事を削除しました", "success")
    return redirect(url_for("admin.admin_dashboard"))


from flask import render_template, request, redirect, url_for, flash, abort, current_app
from flask_login import login_required, current_user
from app.forms import RyunosukeDepositForm
from app.models import User, RyunosukeDeposit, Site, SiteQuotaLog, db
from collections import defaultdict
from datetime import datetime
from sqlalchemy import func, extract, text
import time

@admin_bp.route("/admin/accounting", methods=["GET", "POST"])
@admin_required_effective
def accounting():
    t0 = time.perf_counter()

    selected_month = request.args.get("month", "all")

    # ✅ 入金フォーム処理（POST）
    form = RyunosukeDepositForm()
    if form.validate_on_submit():
        new_deposit = RyunosukeDeposit(
            deposit_date=form.deposit_date.data,
            amount=form.amount.data,
            memo=form.memo.data
        )
        db.session.add(new_deposit)
        db.session.commit()
        flash("龍之介の入金記録を保存しました", "success")
        return redirect(url_for("admin.accounting"))

    # ── 計測開始
    t0 = time.perf_counter()

    # ✅ 入金合計と残高
    paid_total = db.session.query(
        db.func.coalesce(db.func.sum(RyunosukeDeposit.amount), 0)
    ).scalar()
    logger.info("[accounting] t_sum_deposit=%.3f", time.perf_counter()-t0)
    current_app.logger.info("[/admin/accounting] paid_total in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    # ✅ サイト枠合計をSQLひと撃ちで取得（ユーザー配列は未使用なので計算のみ）
    t1 = time.perf_counter()
    row = db.session.execute(text("""
        SELECT
          COALESCE(SUM(CASE
              WHEN u.is_admin = FALSE AND sq.plan_type = 'business' AND sq.total_quota > 0
              THEN sq.total_quota ELSE 0 END), 0) AS business_total,
          COALESCE(SUM(CASE
              WHEN u.is_admin = FALSE AND (u.is_special_access = TRUE OR u.id = 16)
                   AND COALESCE(sq.plan_type, '') <> 'business' AND sq.total_quota > 0
              THEN sq.total_quota ELSE 0 END), 0) AS tcc_1000_total,
          COALESCE(SUM(CASE
              WHEN u.is_admin = FALSE AND (u.is_special_access = FALSE AND u.id <> 16)
                   AND COALESCE(sq.plan_type, '') <> 'business' AND sq.total_quota > 0
              THEN sq.total_quota ELSE 0 END), 0) AS tcc_3000_total
        FROM "user" u
        JOIN user_site_quota sq ON sq.user_id = u.id
    """)).fetchone()
    business_total  = int(row.business_total)
    tcc_1000_total  = int(row.tcc_1000_total)
    tcc_3000_total  = int(row.tcc_3000_total)
    # 画面ではユーザー配列を使っていないため空で渡す（互換維持）
    student_users, member_users, business_users = [], [], []
    current_app.logger.info("[/admin/accounting] load quota sums in %.3fs", time.perf_counter()-t1)


    # ✅ 集計結果（現状の構成は完全維持）
    breakdown = {
        "unpurchased": {
            "count": tcc_3000_total,
            "ryu": tcc_3000_total * 1000,
            "take": tcc_3000_total * 2000,
        },
        "purchased": {
            "count": tcc_1000_total,
            "ryu": 0,
            "take": tcc_1000_total * 1000,
        },
        "business": {
            "count": business_total,
            "ryu": business_total * 16000,
            "take": business_total * 4000,
        },
        "total": {
            "count": tcc_3000_total + tcc_1000_total + business_total,
            "ryu": tcc_3000_total * 1000 + business_total * 16000,
            "take": tcc_3000_total * 2000 + tcc_1000_total * 1000 + business_total * 4000,
        },
    }

    # ✅ サイト登録データを月別にSQLで直接集計（join最適化＋NULL除外）
    t2 = time.perf_counter()
    site_data_raw = (
        db.session.query(
            func.date_trunc("month", Site.created_at).label("month"),
            func.count(Site.id)
        )
        .join(User, Site.user_id == User.id, isouter=False)
        .filter(
            Site.created_at.isnot(None),
            User.is_admin == False,
            User.is_special_access == False  # ← TCC研究生（3,000円）のみ
        )
        .group_by(func.date_trunc("month", Site.created_at))
        .all()
    )
    logger.info("[accounting] t_site_agg=%.3f", time.perf_counter()-t2)
    current_app.logger.info("[/admin/accounting] monthly site agg in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()

    site_data_by_month = {}
    all_months_set = set()

    for month_obj, count in site_data_raw:
        month_key = month_obj.strftime("%Y-%m")
        all_months_set.add(month_key)
        site_data_by_month[month_key] = {
            "site_count": count,
            "ryunosuke_income": count * 1000,
            "takeshi_income": count * 2000
        }

    # ✅ 選択月のみ表示 or 全表示
    filtered_data = (
        site_data_by_month if selected_month == "all"
        else {
            selected_month: site_data_by_month.get(selected_month, {
                "site_count": 0,
                "ryunosuke_income": 0,
                "takeshi_income": 0
            })
        }
    )

    # ✅ 入金履歴と月一覧（変化なし）
    t3 = time.perf_counter()
    deposit_logs = RyunosukeDeposit.query.order_by(RyunosukeDeposit.deposit_date.desc()).all()
    current_app.logger.info("[/admin/accounting] load deposit_logs in %.3fs", time.perf_counter()-t0); t0=time.perf_counter()
    logger.info("[accounting] t_deposits=%.3f", time.perf_counter()-t3)
    all_months = sorted(all_months_set, reverse=True)

    # ✅ テンプレートへ渡す（現状維持）
    t4 = time.perf_counter()
    resp = render_template(
        "admin/accounting.html",
        form=form,
        paid_total=paid_total,
        remaining=breakdown["unpurchased"]["ryu"] - paid_total,
        site_data_by_month=dict(sorted(filtered_data.items())),
        selected_month=selected_month,
        all_months=all_months,
        deposit_logs=deposit_logs,
        breakdown=breakdown,
        student_users=student_users,
        member_users=member_users,
        business_users=business_users
    )
    logger.info("[accounting] t_render=%.3f  t_total=%.3f",
             time.perf_counter()-t4, time.perf_counter()-t0)
    current_app.logger.info("[/admin/accounting] render_template in %.3fs", time.perf_counter()-t0)
    return resp


@admin_bp.route("/admin/accounting/details", methods=["GET"])
@admin_required_effective
def accounting_details():

    selected_month = request.args.get("month", "all")

    # ✅ 月一覧を抽出（NULLを除外して高速に）
    all_months_raw = (
        db.session.query(func.date_trunc("month", SiteQuotaLog.created_at))
        .filter(SiteQuotaLog.created_at.isnot(None))
        .distinct()
        .all()
    )

    all_months = sorted(
        {month[0].strftime("%Y-%m") for month in all_months_raw},
        reverse=True
    )

    # ✅ 月フィルタに応じてログ抽出
    logs_query = SiteQuotaLog.query.filter(SiteQuotaLog.created_at.isnot(None))

    if selected_month != "all":
        try:
            year, month = selected_month.split("-")
            logs_query = logs_query.filter(
                extract("year", SiteQuotaLog.created_at) == int(year),
                extract("month", SiteQuotaLog.created_at) == int(month)
            )
        except Exception:
            flash("不正な月形式です", "error")
            return redirect(url_for("admin.accounting_details"))

    # ✅ 並び順（新しい順）
    logs = logs_query.order_by(SiteQuotaLog.created_at.desc()).all()

    # ✅ テンプレートへ渡す（変化なし）
    return render_template(
        "admin/accounting_details.html",
        logs=logs,
        selected_month=selected_month,
        all_months=all_months
    )


@admin_bp.route("/admin/accounting/adjust", methods=["POST"])
@admin_required_effective
def adjust_quota():

    from flask import request, jsonify

    data = request.get_json()

    try:
        uid = int(data.get("uid"))
        delta = int(data.get("delta", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "uid または delta の形式が不正です"}), 400

    if delta == 0:
        return jsonify({"error": "delta は 0 以外で指定してください"}), 400

    user = User.query.filter_by(id=uid, is_admin=False).first()
    if not user or not user.site_quota:
        return jsonify({"error": "対象ユーザーが見つかりません"}), 404

    quota = user.site_quota
    quota.total_quota = max(quota.total_quota + delta, 0)

    # ログ記録
    quota_log = SiteQuotaLog(
        user_id=user.id,
        plan_type=quota.plan_type,
        site_count=delta,
        reason="管理者手動調整",
        created_at=datetime.utcnow()
    )
    db.session.add(quota_log)
    db.session.commit()

    # ✅ 集計再構築
    stu_cnt = mem_cnt = biz_cnt = 0
    for u in User.query.filter_by(is_admin=False).all():
        sq = u.site_quota
        if not sq or sq.total_quota == 0:
            continue
        if sq.plan_type == "business":
            biz_cnt += sq.total_quota
        elif u.is_special_access:
            mem_cnt += sq.total_quota
        else:
            stu_cnt += sq.total_quota

    PRICES = {
        "student":  {"ryu": 1000,  "take": 2000},
        "member":   {"ryu": 0,     "take": 1000},
        "business": {"ryu": 16000, "take": 4000},
    }

    def calc(cnt, key):
        return {
            "count": cnt,
            "ryu": cnt * PRICES[key]["ryu"],
            "take": cnt * PRICES[key]["take"],
        }

    res_student  = calc(stu_cnt, "student")
    res_member   = calc(mem_cnt, "member")
    res_business = calc(biz_cnt, "business")

    res_total = {
        "count": stu_cnt + mem_cnt + biz_cnt,
        "ryu":   res_student["ryu"] + res_member["ryu"] + res_business["ryu"],
        "take":  res_student["take"] + res_member["take"] + res_business["take"]
    }

    return jsonify({
        "student":  res_student,
        "member":   res_member,
        "business": res_business,
        "total":    res_total,
        "message": f"✅ {user.last_name} {user.first_name} に {delta:+} 件 調整しました"
    })



# --- 既存: ユーザー全記事表示 ---
@admin_bp.route("/admin/user/<int:uid>/articles")
@admin_required_effective
def user_articles(uid):

    from collections import defaultdict
    from app.article_generator import _generate_slots_per_site
    from app.models import User, Article, Site
    from sqlalchemy.orm import selectinload
    from sqlalchemy import asc, nulls_last

    user = User.query.get_or_404(uid)
    status = request.args.get("status")
    sort_key = request.args.get("sort", "scheduled_at")
    sort_order = request.args.get("order", "desc")
    source = request.args.get("source", "all")

    # 🔹 未スケジュールの記事にslot自動割当
    unscheduled = Article.query.filter(
        Article.user_id == user.id,
        Article.scheduled_at.is_(None)
    ).all()

    if unscheduled:
        site_map = defaultdict(list)
        for art in unscheduled:
            if art.site_id:
                site_map[art.site_id].append(art)

        for sid, articles in site_map.items():
            slots = iter(_generate_slots_per_site(current_app, sid, len(articles)))
            for art in articles:
                art.scheduled_at = next(slots)
        db.session.commit()

    # 🔹 記事取得クエリ
    q = Article.query.filter_by(user_id=user.id)
    if status:
        q = q.filter_by(status=status)
    if source == "gsc":
        q = q.filter_by(source="gsc")

    q = q.options(selectinload(Article.site))
    q = q.order_by(nulls_last(asc(Article.scheduled_at)), Article.created_at.desc())
    articles = q.all()

    # 🔽 並び替え（Python側）
    if sort_key == "clicks":
        articles.sort(key=lambda a: a.site.clicks or 0, reverse=(sort_order == "desc"))
    elif sort_key == "impr":
        articles.sort(key=lambda a: a.site.impressions or 0, reverse=(sort_order == "desc"))

    return render_template(
        "admin/user_articles.html",
        articles=articles,
        site=None,
        user=user,
        status=status,
        sort_key=sort_key,
        sort_order=sort_order,
        selected_source=source,
        jst=JST
    )


# --- ✅ 追加: サイト単位の記事一覧表示 ---
@admin_bp.route("/admin/site/<int:site_id>/articles")
@admin_required_effective
def site_articles(site_id):

    from app.models import Site, Article, User
    from sqlalchemy.orm import selectinload
    from sqlalchemy import asc, nulls_last

    site = Site.query.options(selectinload(Site.user)).get_or_404(site_id)
    user = site.user  # ✅ ここで site に紐づく正しい user を取得

    status = request.args.get("status")
    source = request.args.get("source", "all")

    q = Article.query.filter_by(site_id=site.id)
    if status:
        q = q.filter_by(status=status)
    if source == "gsc":
        q = q.filter_by(source="gsc")

    q = q.options(selectinload(Article.site))
    q = q.order_by(nulls_last(asc(Article.scheduled_at)), Article.created_at.desc())
    articles = q.all()

    return render_template(
        "admin/user_articles.html",
        articles=articles,
        site=site,
        user=user,  # ✅ この user は site に紐づいたもの
        status=status,
        sort_key=None,
        sort_order=None,
        selected_source=source,
        jst=JST
    )



@admin_bp.post("/admin/user/<int:uid>/delete-stuck")
@admin_required_effective
def delete_user_stuck_articles(uid):

    user = User.query.get_or_404(uid)

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).all()

    count = len(stuck_articles)
    for art in stuck_articles:
        db.session.delete(art)
    db.session.commit()

    flash(f"{count} 件の途中停止記事を削除しました", "success")
    return redirect(url_for("admin.user_articles", uid=uid))

@admin_bp.post("/admin/login-as/<int:user_id>")
@admin_required_effective
def admin_login_as(user_id):
    # 有効管理者のチェック（通常管理者 or 既にadmin_id保持中）

    # いま本当に管理者としてログインしている場合、元の管理者IDを保持
    # （既に保持しているなら上書きしない＝多段なりすましを避ける）
    if ("admin_id" not in session) and getattr(current_user, "is_admin", False):
        session["admin_id"] = current_user.id

    # 対象ユーザーに完全切替（＝以後 current_user は対象ユーザー）
    user = User.query.get_or_404(user_id)
    login_user(user)

    flash(f"{user.email} としてログインしました（管理者モード維持）", "info")
    return redirect(url_for("main.dashboard", username=user.username))


@admin_bp.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required_effective
def delete_user(user_id):

    user = User.query.get_or_404(user_id)

    db.session.delete(user)
    db.session.commit()

    flash("✅ ユーザーと関連データをすべて削除しました。", "success")
    return redirect(url_for("admin.admin_users"))


# ──────────────── GSCサイト状況一覧（管理者）────────────────
@admin_bp.route("/admin/gsc_sites")
@admin_required_effective
def admin_gsc_sites():

    from sqlalchemy.orm import selectinload
    from collections import defaultdict
    from app.models import Site, User, Keyword, Article

    # 全サイトをユーザー単位で取得（リレーション付きで最適化）
    users = User.query.options(selectinload(User.sites)).all()

    user_site_data = []

    for user in users:
        site_infos = []
        for site in user.sites:
            if not site.gsc_connected:
                continue  # GSC未接続サイトは除外

            # GSCキーワード全件
            keywords = Keyword.query.filter_by(site_id=site.id, source="gsc").all()
            done        = sum(1 for k in keywords if k.status == "done")
            generating  = sum(1 for k in keywords if k.status == "generating")
            unprocessed = sum(1 for k in keywords if k.status == "unprocessed")

            # 最新取得・生成日
            latest_keyword_date = max([k.created_at for k in keywords], default=None)

            # GSC記事の最新生成日（Article参照）
            latest_article = Article.query.filter_by(site_id=site.id, source="gsc").order_by(Article.created_at.desc()).first()
            latest_article_date = latest_article.created_at if latest_article else None

            site_infos.append({
                "site": site,
                "done": done,
                "generating": generating,
                "unprocessed": unprocessed,
                "total": done + generating + unprocessed,
                "latest_keyword_date": latest_keyword_date,
                "latest_article_date": latest_article_date
            })

        if site_infos:
            user_site_data.append({
                "user": user,
                "sites": site_infos
            })

    return render_template("admin/gsc_sites.html", user_site_data=user_site_data)


# ──────────────── NEW: インデックス進捗モニター（閲覧専用）────────────────
@admin_bp.route("/admin/index_monitor")
@login_required
def admin_index_monitor():
    """全ユーザー・全サイトのインデックス率を高速集計して表示"""
    from datetime import date, timedelta
    from app.models import Site, Article, GSCDailyTotal, User

    # ✅ 直近28日の窓を統一（JSTの昨日 ∧ DB最新日）
    start_d, end_d = _gsc_window_by_latest_db(28)

    # 🔹 直近28日間の GSC掲載データ集計（site単位）
    sub_gsc = (
        db.session.query(
            GSCDailyTotal.site_id,
            func.count(GSCDailyTotal.id).label("indexed_count")
        )
        .filter(GSCDailyTotal.date >= start_d, GSCDailyTotal.date <= end_d)
        .group_by(GSCDailyTotal.site_id)
        .subquery()
    )

    # 🔹 サイトごとの記事数＋インデックス件数
    results = (
        db.session.query(
            Site.id, Site.url, Site.user_id,
            User.username,
            func.count(Article.id).label("article_count"),
            func.coalesce(sub_gsc.c.indexed_count, 0).label("indexed_count")
        )
        .join(User, User.id == Site.user_id)
        .outerjoin(Article, Article.site_id == Site.id)
        .outerjoin(sub_gsc, sub_gsc.c.site_id == Site.id)
        .group_by(Site.id, User.username, sub_gsc.c.indexed_count)
        .order_by(func.coalesce(sub_gsc.c.indexed_count, 0).asc())  # インデックス少ない順
        .limit(50)  # 速度重視
        .all()
    )

    # 🔹 表示用に整形
    data = []
    for site_id, url, user_id, username, total, indexed in results:
        rate = (indexed / total * 100) if total else 0
        data.append({
            "url": url,
            "username": username,
            "article_count": total,
            "indexed_count": indexed,
            "rate": round(rate, 1),
        })

    return render_template("admin/index_monitor.html", data=data)


@admin_bp.get("/admin/user/<int:uid>/stuck-articles")
@admin_required_effective
def stuck_articles(uid):

    user = User.query.get_or_404(uid)

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).order_by(Article.created_at.desc()).all()

    return render_template("admin/stuck_articles.html", user=user, articles=stuck_articles)


@admin_bp.post("/admin/user/<int:uid>/regenerate-stuck")
@admin_required_effective
def regenerate_user_stuck_articles(uid):

    stuck_articles = Article.query.filter(
        Article.user_id == uid,
        Article.status.in_(["pending", "gen"])
    ).all()

    app = current_app._get_current_object()

    def _background_regeneration():
        with app.app_context():
            prompt = PromptTemplate.query.filter_by(user_id=uid).first()
            if not prompt:
                logging.warning(f"[再生成中止] user_id={uid} にプロンプトがありません")
                return

            for art in stuck_articles:
                try:
                    threading.Thread(
                        target=_generate,
                        args=(app, art.id, prompt.title_pt, prompt.body_pt),
                        daemon=True
                    ).start()
                except Exception as e:
                    logging.exception(f"[管理者再生成失敗] article_id={art.id} error={e}")

    threading.Thread(target=_background_regeneration, daemon=True).start()

    flash(f"{len(stuck_articles)} 件の途中停止記事を再生成キューに登録しました（バックグラウンド処理）", "success")
    return redirect(url_for("admin.stuck_articles", uid=uid))



# 先頭の import を修正
# 既存の import に追加（上の方）
from flask import Blueprint, request, jsonify, Response, redirect, url_for, render_template, current_app
from flask_login import login_required, current_user
from sqlalchemy import func, desc, asc, and_
from datetime import datetime, timedelta, timezone
from app import db
from app.models import User, Site, Article, GSCDailyTotal

# ────────────────────────────────────────────────
# GSC 28日などの集計窓を統一する極小ヘルパー
# ・終端は「JSTの昨日」と「DBの最新日」の早い方
# ・返り値: (start_date, end_date) いずれも date 型（両端含む）
# ────────────────────────────────────────────────
def _gsc_window_by_latest_db(days: int = 28):
    from app import db
    from app.models import GSCDailyTotal
    JST = timezone(timedelta(hours=9))
    today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
    end_by_yesterday = today_jst - timedelta(days=1)
    latest_db_date = db.session.query(func.max(GSCDailyTotal.date)).scalar()
    if latest_db_date:
        end_date = min(end_by_yesterday, latest_db_date)
    else:
        end_date = end_by_yesterday
    start_date = end_date - timedelta(days=max(1, int(days)) - 1)
    return start_date, end_date


# ← これを先頭の import セクションに追加
from app.utils.monitor import (
    get_memory_usage,
    get_cpu_load,
    get_latest_restart_log,
    get_last_restart_time,
)
import json

# ※ admin_bp は既存の Blueprint を使用

@admin_bp.route("/api/admin/rankings")
@admin_required_effective
def admin_rankings():

    # ==== クエリ取得 ====
    rank_type = (request.args.get("type") or "site").lower()        # site / impressions / clicks / posted_articles
    order     = (request.args.get("order") or "desc").lower()       # asc / desc
    period    = (request.args.get("period") or "3m").lower()        # 1d / 7d / 28d / 3m / 6m / 12m / 16m / custom / all
    start_str = request.args.get("start_date")
    end_str   = request.args.get("end_date")

    sort_func = asc if order == "asc" else desc

    # ==== JST日付の境界を作る（GSCに合わせる） ====
    JST = timezone(timedelta(hours=9))
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    now_jst = now_utc.astimezone(JST)

    # プリセット → JST基準の開始日時を決定
    def jst_date(d: datetime) -> datetime.date:
        return d.astimezone(JST).date()

    presets = {
        "1d":  now_jst - timedelta(days=1),
        "7d":  now_jst - timedelta(days=7),
        "28d": now_jst - timedelta(days=28),
        "3m":  now_jst - timedelta(days=90),
        "6m":  now_jst - timedelta(days=180),
        "12m": now_jst - timedelta(days=365),
        "16m": now_jst - timedelta(days=480),
        "all": None,
    }

    # 期間決定（JST日付で保持）
    latest_db_date = db.session.query(func.max(GSCDailyTotal.date)).scalar()
    if period == "custom":
        try:
            # customは yyyy-mm-dd（ローカル=JST想定）をそのまま日付として使う
            start_jst_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
            end_jst_date   = datetime.strptime(end_str, "%Y-%m-%d").date()   if end_str   else jst_date(now_jst)
            # ✅ GSC集計（impressions/clicks）は「昨日締め」に丸める
            if rank_type in ("impressions", "clicks") and end_jst_date >= jst_date(now_jst):
                end_jst_date = end_jst_date - timedelta(days=1)
            # ✅ DB最新日にクランプ（未取得・未確定日の除外）
            if rank_type in ("impressions", "clicks") and latest_db_date:
                if end_jst_date and latest_db_date < end_jst_date:
                    end_jst_date = latest_db_date
                # start未指定や start>end の場合は28日窓を補完
                if (not start_jst_date) or (start_jst_date > end_jst_date):
                    start_jst_date = end_jst_date - timedelta(days=27)    
        except ValueError:
            return jsonify({"error": "日付形式が不正です (YYYY-MM-DD)"}), 400
    else:
        if period == "all":
            start_jst_date, end_jst_date = None, None
        else:
            start_dt_jst = presets.get(period, now_jst - timedelta(days=90))  # デフォは3か月相当
            # ✅ GSC集計（impressions/clicks）は昨日で締める
            if rank_type in ("impressions", "clicks"):
                end_dt_jst   = now_jst - timedelta(days=1)
                # DB最新日でクランプ
                if latest_db_date and latest_db_date < jst_date(end_dt_jst):
                    end_dt_jst = datetime.combine(latest_db_date, datetime.min.time(), tzinfo=JST)
                start_jst_date = jst_date(start_dt_jst if start_dt_jst < end_dt_jst else end_dt_jst)
                end_jst_date   = jst_date(end_dt_jst)
            else:
                start_jst_date = jst_date(start_dt_jst)
                end_jst_date   = jst_date(now_jst)

    try:
        # ====== 1) サイト数（総数）======
        if rank_type == "site":
            subq = (
                db.session.query(
                    User.id.label("user_id"),
                    User.last_name,
                    User.first_name,
                    func.count(Site.id).label("site_count")
                )
                .outerjoin(Site, Site.user_id == User.id)
                .group_by(User.id, User.last_name, User.first_name)
                .subquery()
            )
            rows = (
                db.session.query(subq.c.last_name, subq.c.first_name, subq.c.site_count)
                .order_by(sort_func(subq.c.site_count))
                .all()
            )
            data = [{"last_name": r.last_name, "first_name": r.first_name, "site_count": int(r.site_count or 0)} for r in rows]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        # ====== 2) 表示回数 / クリック数：GSCMetricから期間合算 ======
                # ====== 2) 表示回数 / クリック数：GSCDailyTotal から期間SUM ======
        elif rank_type in ("impressions", "clicks"):
            metric_col = (
                func.coalesce(func.sum(GSCDailyTotal.impressions), 0)
                if rank_type == "impressions"
                else func.coalesce(func.sum(GSCDailyTotal.clicks), 0)
            ).label("value")

            # 期間は JST の日付（start_jst_date / end_jst_date）が既に決まっている
            # GSCDailyTotal.date は Date カラムなので、そのまま inclusive でOK
            join_on = and_(
                GSCDailyTotal.site_id == Site.id,
                (GSCDailyTotal.date >= start_jst_date) if start_jst_date else True,
                (GSCDailyTotal.date <= end_jst_date) if end_jst_date else True,
            )

            q = (
                db.session.query(
                    Site.id.label("site_id"),
                    Site.name.label("site_name"),
                    Site.url.label("site_url"),
                    User.last_name,
                    User.first_name,
                    metric_col,
                )
                .join(User, Site.user_id == User.id)
                .outerjoin(GSCDailyTotal, join_on)
                .group_by(Site.id, Site.name, Site.url, User.last_name, User.first_name)
                .order_by(sort_func(metric_col))
            )

            rows = q.all()
            data = [
                {
                    "site_name": r.site_name,
                    "site_url": r.site_url,
                    "user_name": f"{r.last_name} {r.first_name}",
                    "value": int(r.value or 0),
                }
                for r in rows
            ]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")


        # ====== 3) 投稿完了記事数：posted_at をJST期間で計上 ======
        elif rank_type == "posted_articles":
            # JST日付 → UTCの境界に変換（JST 00:00:00 をUTCに直す）
            def jst_date_to_utc_start(d: datetime.date) -> datetime:
                # JST 00:00 -> UTC前日 15:00
                return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=JST).astimezone(timezone.utc)

            def jst_date_to_utc_end(d: datetime.date) -> datetime:
                # JST 23:59:59.999 -> 翌日JST 00:00 の直前 = UTC同日 14:59:59.999...
                nxt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=JST) + timedelta(days=1)
                return nxt.astimezone(timezone.utc)

            q = (
                db.session.query(
                    Site.name.label("site_name"),
                    Site.url.label("site_url"),
                    User.last_name,
                    User.first_name,
                    func.count(Article.id).label("value")
                )
                .join(User, Site.user_id == User.id)
                .join(Article, Article.site_id == Site.id)
                .filter(Article.status == "posted")
            )

            if start_jst_date:
                q = q.filter(Article.posted_at >= jst_date_to_utc_start(start_jst_date))
            if end_jst_date:
                q = q.filter(Article.posted_at <  jst_date_to_utc_end(end_jst_date))  # endは翌日0時未満で閉区間相当

            q = (
                q.group_by(Site.id, Site.name, Site.url, User.last_name, User.first_name)
                 .order_by(sort_func(func.count(Article.id)))
            )
            rows = q.all()
            data = [{
                "site_name": r.site_name,
                "site_url": r.site_url,
                "user_name": f"{r.last_name} {r.first_name}",
                "value": int(r.value or 0),
            } for r in rows]
            return Response(json.dumps(data, ensure_ascii=False), mimetype="application/json")

        else:
            return jsonify({"error": "不正なランキングタイプです"}), 400

    except Exception as e:
        current_app.logger.exception("[admin_rankings] server error")
        return jsonify({"error": "server_error", "detail": str(e)}), 500


@admin_bp.route("/admin/ranking-page")
@admin_required_effective
def admin_ranking_page():
    if not getattr(current_user, "is_admin", False):
        return redirect(url_for("main.dashboard", username=current_user.username))
    return render_template("admin/ranking_page.html")



# 監視ページ
@admin_bp.route("/admin/monitoring")
@admin_required_effective
def admin_monitoring():

    memory = get_memory_usage()
    cpu = get_cpu_load()
    restart_logs = get_latest_restart_log()
    last_restart = get_last_restart_time()

    return render_template("admin/monitoring.html",
                           memory=memory,
                           cpu=cpu,
                           restart_logs=restart_logs,
                           last_restart=last_restart)


@admin_bp.route("/admin/captcha-dataset", methods=["POST"])
@admin_required_effective
def admin_captcha_label_update():
    from pathlib import Path

    image_file = request.form.get("image_file")
    new_label = request.form.get("label", "").strip()

    if not image_file or not new_label:
        return "無効な入力", 400

    dataset_dir = Path("data/captcha_dataset")
    txt_path = dataset_dir / Path(image_file).with_suffix(".txt")

    try:
        txt_path.write_text(new_label, encoding="utf-8")
        flash(f"{image_file} のラベルを更新しました。", "success")
    except Exception as e:
        flash(f"ラベル更新失敗: {e}", "danger")

    return redirect(url_for("admin.admin_captcha_dataset"))

@admin_bp.route("/admin/captcha-dataset", methods=["GET"])
@admin_required_effective
def admin_captcha_dataset():
    from pathlib import Path
    from flask import render_template

    # ✅ 学習用データ
    dataset_dir = Path("data/captcha_dataset")
    dataset_entries = []
    for path in sorted(dataset_dir.glob("*.png")):
        label_path = path.with_suffix(".txt")
        label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        dataset_entries.append({
            "image_file": path.name,
            "image_url": url_for('static', filename=f"../data/captcha_dataset/{path.name}"),
            "label": label
        })

    # ✅ 本番保存失敗画像（app/static/captchas）
    captchas_dir = Path("app/static/captchas")
    captcha_entries = []
    for path in sorted(captchas_dir.glob("*.png")):
        captcha_entries.append({
            "image_file": path.name,
            "image_url": url_for('static', filename=f"captchas/{path.name}"),
            "label": "（未設定）"
        })

    # ✅ 結合してテンプレートへ渡す
    entries = dataset_entries + captcha_entries

    return render_template("admin/captcha_dataset.html", entries=entries)

# 内部SEOルートコード

# 内部SEOルートコード（admin_bp 配下）
import os
from datetime import datetime, timedelta, timezone
from flask import render_template, request, redirect, url_for, flash, abort, jsonify, make_response
from flask_login import login_required, current_user
from sqlalchemy import desc, and_, or_, func
from sqlalchemy.orm import load_only, defer
from sqlalchemy import text

from app import db
from app.models import (
    Site,
    InternalSeoRun,
    InternalLinkAction,
    ContentIndex,
    User,
    InternalSeoUserSchedule,
    InternalSeoUserRun,
)
from sqlalchemy import func, and_, desc, text

# 🆕 内部SEOサービス（planner / applier）
from app.services.internal_seo.applier import (
    preview_apply_for_post,
    apply_actions_for_post,
)
from app.services.internal_seo.planner import (
    plan_links_for_post,
)

# 🆕 ユーザー単位スケジューラ（サービス層）
try:
    # 先に作成した app/services/internal_seo/user_scheduler.py
    from app.services.internal_seo.user_scheduler import (
        enqueue_user_tick,
        run_user_tick,  # run_once 用
    )
except Exception:
    # 開発中でも routes の import で落ちないように保険
    enqueue_user_tick = None
    run_user_tick = None

JST = timezone(timedelta(hours=9))


# ---- stats: 1ラン分の詳細 ----
@admin_bp.route("/admin/internal-seo/run/<int:run_id>/stats", methods=["GET"])
@admin_required_effective
def admin_internal_seo_run_stats(run_id: int):
    
    run = InternalSeoRun.query.get_or_404(run_id)
    payload = {"ok": True, "stats": run.stats or {}}
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "public, max-age=30"
    return resp

# ---- 画面本体 ----
@admin_bp.route("/admin/internal-seo", methods=["GET"])
@admin_required_effective
def admin_internal_seo_index():

    # ダッシュボード（KPI + 進捗 + リアルタイムログ）
    days = request.args.get("days", default=7, type=int)
    return render_template("admin/internal_seo.html", days=days)

# ---- 概要ダッシュボード（総数 / 適用済み / キュー / 直近ラン）----
@admin_bp.route("/admin/internal-seo/overview", methods=["GET"])
@admin_required_effective
def admin_internal_seo_overview():
    

    # 総サイト数
    total_sites = Site.query.count()

    # 1件以上 "applied" の内部リンクがあるサイト数
    applied_sites = (
        db.session.query(InternalLinkAction.site_id)
        .filter(InternalLinkAction.status == "applied")
        .distinct()
        .count()
    )

    # ジョブキュー状況
    rows = db.session.execute(
        text("SELECT status, COUNT(*) AS cnt FROM internal_seo_job_queue GROUP BY status")
    ).mappings().all()
    queue_summary = {r["status"]: int(r["cnt"]) for r in rows}

    # 直近ラン（20件）
    recent_runs = (
        InternalSeoRun.query
        .order_by(InternalSeoRun.id.desc())
        .limit(20)
        .all()
    )

    return render_template(
        "admin/internal_seo_overview.html",
        total_sites=total_sites,
        applied_sites=applied_sites,
        queue_summary=queue_summary,
        recent_runs=recent_runs,
    )


@admin_bp.route("/admin/internal-seo/preview", methods=["GET"])
def admin_internal_seo_preview():
    """
    ドライラン（実適用なし）で、どの語句がどのURLにリンクされるかのプレビューを返す。
    ?site_id=...&post_id=...&format=json
    """
    site_id = request.args.get("site_id", type=int)
    post_id = request.args.get("post_id", type=int)
    fmt = (request.args.get("format") or "json").lower()
    if not site_id or not post_id:
        return jsonify({"ok": False, "error": "missing site_id or post_id"}), 400

    html, res, items = preview_apply_for_post(site_id, post_id)

    if fmt == "json":
        return jsonify({
            "ok": True,
            "result": {
                "applied": res.applied,
                "swapped": res.swapped,
                "skipped": res.skipped,
                "message": res.message,
            },
            "previews": [
                {
                    "position": it.position,
                    "anchor_text": it.anchor_text,
                    "target_post_id": it.target_post_id,
                    "target_url": it.target_url,
                    "paragraph_index": it.paragraph_index,
                    "paragraph_excerpt_before": it.paragraph_excerpt_before,
                    "paragraph_excerpt_after": it.paragraph_excerpt_after,
                }
                for it in items
            ],
        })
    elif fmt == "html":
        # HTMLビュー（テンプレートは次ステップで追加）
        return render_template(
            "admin/internal_seo_preview.html",
            site_id=site_id,
            post_id=post_id,
            result=res,
            previews=items,
        )
    else:
        return jsonify({"ok": False, "error": "unsupported format"}), 400
    
# ---- 🆕 現役バージョン / 世代集計（ポスト単位） ----
@admin_bp.route("/admin/internal-seo/post/<int:post_id>/versions", methods=["GET"])
@admin_required_effective
def admin_internal_seo_post_versions(post_id: int):
    """
    現在の post_id について、link_version の分布と現役（max applied）を返す。
    """
    site_row = (
        ContentIndex.query
        .with_entities(ContentIndex.site_id)
        .filter(ContentIndex.wp_post_id == post_id)
        .one_or_none()
    )
    if not site_row:
        return jsonify({"ok": False, "error": "post not found"}), 404
    site_id = int(site_row[0])

    # バージョン分布
    dist_rows = (
        db.session.query(
            InternalLinkAction.link_version,
            InternalLinkAction.status,
            func.count(InternalLinkAction.id),
        )
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
        )
        .group_by(InternalLinkAction.link_version, InternalLinkAction.status)
        .all()
    )
    dist = {}
    for ver, st, cnt in dist_rows:
        v = int(ver or 0)
        dist.setdefault(v, {})
        dist[v][st] = int(cnt or 0)

    # 現役（applied の最大 version）
    current_row = (
        db.session.query(func.max(InternalLinkAction.link_version))
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
            InternalLinkAction.status == "applied",
        )
        .one()
    )
    current_version = int(current_row[0] or 0)
    return jsonify({"ok": True, "site_id": site_id, "post_id": post_id, "current_version": current_version, "distribution": dist})


# ---- 🆕 再ビルド（計画のみ） ----
@admin_bp.route("/admin/internal-seo/rebuild/plan", methods=["POST"])
@admin_required_effective
def admin_internal_seo_rebuild_plan():
    """
    全置換ルール:
      - 旧 max(link_version) を特定
      - 旧 'applied' を 'reverted' + reverted_at=now に更新（履歴保持）
      - 新規 'pending' を作成し、link_version = 旧max + 1 を付与（plannerで生成後に付与）
    返却: 新規 pending 件数と新version
    """
    post_id = request.form.get("post_id", type=int) or (request.get_json(silent=True) or {}).get("post_id")
    if not post_id:
        return jsonify({"ok": False, "error": "post_id required"}), 400

    # post から site_id を解決
    ci = (
        ContentIndex.query
        .with_entities(ContentIndex.site_id)
        .filter(ContentIndex.wp_post_id == post_id)
        .one_or_none()
    )
    if not ci:
        return jsonify({"ok": False, "error": "post not found"}), 404
    site_id = int(ci[0])

    now = datetime.utcnow()
    # 旧 max バージョン
    old_max_row = (
        db.session.query(func.max(InternalLinkAction.link_version))
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
            InternalLinkAction.status.in_(["applied", "skipped", "pending", "reverted", "legacy_deleted"]),
        )
        .one()
    )
    old_max = int(old_max_row[0] or 0)
    new_version = old_max + 1

    # 既存 applied を reverted に（履歴は残す）
    db.session.query(InternalLinkAction)\
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
            InternalLinkAction.status == "applied",
        )\
        .update(
            {
                InternalLinkAction.status: "reverted",
                InternalLinkAction.reverted_at: now,
                InternalLinkAction.updated_at: now,
            },
            synchronize_session=False,
        )
    # 既存 pending は一旦掃除（完全置換のため）
    db.session.query(InternalLinkAction)\
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
            InternalLinkAction.status == "pending",
        ).delete(synchronize_session=False)
    db.session.commit()

    # planner で新規 pending を作成（位置は h2:* 仕様）
    st = plan_links_for_post(
        site_id=site_id,
        src_post_id=post_id,
        mode_swap_check=False,  # 再ビルド時は swap 候補は不要
    )

    # 直近作成の pending に新 version を付与
    pending_q = (
        db.session.query(InternalLinkAction)
        .filter(
            InternalLinkAction.site_id == site_id,
            InternalLinkAction.post_id == post_id,
            InternalLinkAction.status == "pending",
        )
    )
    new_pending = pending_q.all()
    for a in new_pending:
        a.link_version = new_version
        a.updated_at = now
    db.session.commit()

    return jsonify({
        "ok": True,
        "site_id": site_id,
        "post_id": post_id,
        "planned": int(st.planned_actions or 0),
        "new_version": new_version,
    })


# ---- 🆕 再ビルド（適用まで一気に） ----
@admin_bp.route("/admin/internal-seo/rebuild/apply", methods=["POST"])
@admin_required_effective
def admin_internal_seo_rebuild_apply():
    """
    plan と同じ手順で世代を進めた上で、applier を実行。
    フラグ apply=true の簡易版として分離。
    """
    post_id = request.form.get("post_id", type=int) or (request.get_json(silent=True) or {}).get("post_id")
    if not post_id:
        return jsonify({"ok": False, "error": "post_id required"}), 400

    # まず plan（上の関数を内部呼び出ししてもいいが、同ロジックを軽く再実装）
    plan_resp = admin_internal_seo_rebuild_plan()
    if isinstance(plan_resp, tuple):
        payload, code = plan_resp
        if code != 200:
            return plan_resp
        plan_data = payload.get_json() if hasattr(payload, "get_json") else {}
    else:
        plan_data = plan_resp.get_json() if hasattr(plan_resp, "get_json") else {}
    if not (plan_data or {}).get("ok"):
        return plan_resp

    # site_id 解決
    ci = (
        ContentIndex.query
        .with_entities(ContentIndex.site_id)
        .filter(ContentIndex.wp_post_id == post_id)
        .one_or_none()
    )
    site_id = int(ci[0]) if ci else None
    if not site_id:
        return jsonify({"ok": False, "error": "post not found"}), 404

    # applier 実行
    res = apply_actions_for_post(site_id, post_id, dry_run=False)
    return jsonify({
        "ok": True,
        "site_id": site_id,
        "post_id": post_id,
        "applied": int(res.applied or 0),
        "swapped": int(res.swapped or 0),
        "skipped": int(res.skipped or 0),
        "legacy_deleted": int(getattr(res, "legacy_deleted", 0) or 0),
        "message": res.message or "",
        "new_version": plan_data.get("new_version"),
    })    
    

# ---- 進捗（ユーザー × サイト） ----
@admin_bp.route("/admin/internal-seo/progress", methods=["GET"])
@admin_required_effective
def admin_internal_seo_progress():
    """
    各ユーザー × 各サイトの進捗（期間内）
    - last_run（直近処理日時）
    - applied_links / skipped / removed_in_headings / legacy_removed（期間合計）
    - queue_status（queued/running/idle）
    """
    days = int(request.args.get("days", 7))
    since = (datetime.now(JST) - timedelta(days=days)).astimezone(timezone.utc)

    sql = text("""
      WITH site_info AS (
        SELECT s.id AS site_id, s.name AS site_name, s.user_id
        FROM site s
      ),
      user_info AS (
        SELECT u.id AS user_id, u.username
        FROM "user" u
      ),
      logs AS (
        SELECT
          l.site_id,
          l.status,
          COALESCE(l.applied_links, (l.details->>'applied_links')::int) AS applied_links,
          COALESCE(l.removed_in_headings, (l.details->>'removed_in_headings')::int) AS removed_in_headings,
          COALESCE(l.legacy_removed, (l.details->>'legacy_removed')::int) AS legacy_removed,
          l.created_at
        FROM internal_seo_job_log l
        WHERE l.created_at >= :since
      ),
      last_run AS (
        SELECT site_id, MAX(created_at) AS last_run_at
        FROM logs
        GROUP BY site_id
      ),
      agg AS (
        SELECT
          site_id,
          COALESCE(SUM(CASE WHEN status='applied' THEN applied_links ELSE 0 END),0) AS applied_links,
          COALESCE(SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END),0) AS skipped_count,
          COALESCE(SUM(removed_in_headings),0) AS removed_in_headings,
          COALESCE(SUM(legacy_removed),0)       AS legacy_removed
        FROM logs
        GROUP BY site_id
      ),
      qstat AS (
        SELECT site_id,
               MAX(CASE WHEN status='running' THEN 2
                        WHEN status='queued'  THEN 1
                        ELSE 0 END) AS st_rank
        FROM internal_seo_job_queue
        GROUP BY site_id
      )
      SELECT
        ui.user_id, ui.username,
        si.site_id, si.site_name,
        lr.last_run_at,
        COALESCE(a.applied_links,0)      AS applied_links,
        COALESCE(a.skipped_count,0)      AS skipped_count,
        COALESCE(a.removed_in_headings,0) AS removed_in_headings,
        COALESCE(a.legacy_removed,0)      AS legacy_removed,
        CASE COALESCE(q.st_rank,0)
          WHEN 2 THEN 'running'
          WHEN 1 THEN 'queued'
          ELSE 'idle'
        END AS queue_status
      FROM site_info si
      JOIN user_info ui ON ui.user_id = si.user_id
      LEFT JOIN last_run lr ON lr.site_id = si.site_id
      LEFT JOIN agg a      ON a.site_id  = si.site_id
      LEFT JOIN qstat q    ON q.site_id  = si.site_id
      ORDER BY ui.username ASC, si.site_name ASC
    """)
    rows = db.session.execute(sql, {"since": since}).mappings().all() or []

    data = []
    for r in rows:
        data.append({
            "user_id": r.get("user_id"),
            "username": r.get("username"),
            "site_id": r.get("site_id"),
            "site_name": r.get("site_name"),
            "last_run_at": (r.get("last_run_at").astimezone(JST).isoformat(timespec="seconds")
                            if r.get("last_run_at") else None),
            "applied_links": int(r.get("applied_links") or 0),
            "skipped_count": int(r.get("skipped_count") or 0),
            "removed_in_headings": int(r.get("removed_in_headings") or 0),
            "legacy_removed": int(r.get("legacy_removed") or 0),
            "queue_status": r.get("queue_status") or "idle",
        })
    return jsonify({"days": days, "rows": data})
    

# ---- NEW: オーナー一覧（ユーザー別セクション） ----
@admin_bp.route("/admin/internal-seo/owners", methods=["GET"])
@admin_required_effective
def admin_internal_seo_owners():
    

    # Site.owner_id または Site.user_id を優先採用
    owner_col = getattr(Site, "owner_id", None) or getattr(Site, "user_id", None)

    if owner_col is None:
        total_sites = db.session.query(func.count(Site.id)).scalar() or 0
        running_count = (
            db.session.query(func.count(func.distinct(InternalSeoRun.site_id)))
            .filter(InternalSeoRun.status == "running")
            .scalar() or 0
        )
        payload = {"ok": True, "rows": [
            {"id": None, "name": "全サイト", "site_count": int(total_sites), "running_count": int(running_count)}
        ]}
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "public, max-age=30"
        return resp

    site_counts = (
        db.session.query(owner_col.label("owner_id"), func.count(Site.id).label("cnt"))
        .group_by(owner_col)
        .all()
    )
    running_counts = {
        owner_id: cnt
        for owner_id, cnt in (
            db.session.query(owner_col.label("owner_id"), func.count(func.distinct(InternalSeoRun.site_id)))
            .join(Site, Site.id == InternalSeoRun.site_id)
            .filter(InternalSeoRun.status == "running")
            .group_by(owner_col)
            .all()
        )
    }

    rows = []
    for r in site_counts:
        oid = r.owner_id
        rows.append(dict(
            id=(int(oid) if oid is not None else None),
            name=f"ユーザー {oid}" if oid is not None else "全サイト",
            site_count=int(r.cnt or 0),
            running_count=int(running_counts.get(oid, 0)),
        ))

    payload = {"ok": True, "rows": rows}
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "public, max-age=30"
    return resp

# ---- サイト一覧（owner_id / 検索 / カーソル / メトリクス付き） ----
# GET /admin/internal-seo/sites?q=&owner_id=&limit=&cursor_id=
@admin_bp.route("/admin/internal-seo/sites", methods=["GET"])
@admin_required_effective
def admin_internal_seo_sites():

    q = (request.args.get("q") or "").strip()
    owner_id = request.args.get("owner_id", type=int)
    limit = min(max(request.args.get("limit", type=int, default=24), 1), 200)
    cursor_id = request.args.get("cursor_id", type=int)

    base_q = Site.query
    owner_col = getattr(Site, "owner_id", None) or getattr(Site, "user_id", None)
    if owner_col is not None and owner_id is not None:
        base_q = base_q.filter(owner_col == owner_id)

    if q:
        if q.isdigit():
            base_q = base_q.filter(or_(Site.id == int(q), Site.name.ilike(f"%{q}%")))
        else:
            base_q = base_q.filter(Site.name.ilike(f"%{q}%"))

    if cursor_id:
        base_q = base_q.filter(Site.id > cursor_id)

    try:
        base_q = base_q.options(load_only(Site.id, Site.name))
    except Exception:
        base_q = base_q.options(load_only(Site.id))

    base_q = base_q.order_by(Site.id.asc())
    sites = base_q.limit(limit).all()

    site_ids = [s.id for s in sites]
    metrics_map = {}

    if site_ids:
        # InternalLinkAction のステータス別集計
        status_counts = (
            db.session.query(
                InternalLinkAction.site_id,
                InternalLinkAction.status,
                func.count(InternalLinkAction.id),
                func.max(InternalLinkAction.applied_at),
            )
            .filter(InternalLinkAction.site_id.in_(site_ids))
            .group_by(InternalLinkAction.site_id, InternalLinkAction.status)
            .all()
        )
        for sid in site_ids:
            metrics_map[sid] = dict(applied=0, pending=0, skipped=0, last_applied_at=None)

        for sid, st, cnt, max_applied in status_counts:
            if st == "applied":
                metrics_map[sid]["applied"] = int(cnt or 0)
                metrics_map[sid]["last_applied_at"] = max_applied
            elif st == "pending":
                metrics_map[sid]["pending"] = int(cnt or 0)
            elif st == "skipped":
                metrics_map[sid]["skipped"] = int(cnt or 0)

        # 実行中サイト
        running_sites = {
            x[0] for x in db.session.query(InternalSeoRun.site_id)
            .filter(InternalSeoRun.site_id.in_(site_ids), InternalSeoRun.status == "running")
            .group_by(InternalSeoRun.site_id).all()
        }

        # サイト毎の最新Run
        sub = (
            db.session.query(
                InternalSeoRun.site_id.label("sid"),
                func.max(InternalSeoRun.started_at).label("mx")
            )
            .filter(InternalSeoRun.site_id.in_(site_ids))
            .group_by(InternalSeoRun.site_id)
        ).subquery()

        last_runs = (
            db.session.query(
                InternalSeoRun.site_id,
                InternalSeoRun.status,
                InternalSeoRun.started_at,
                InternalSeoRun.ended_at,
                InternalSeoRun.duration_ms,
            )
            .join(sub, and_(InternalSeoRun.site_id == sub.c.sid, InternalSeoRun.started_at == sub.c.mx))
            .all()
        )

        for sid in site_ids:
            m = metrics_map.setdefault(sid, {})
            m["running"] = sid in running_sites

        for sid, st, st_at, ed_at, dur in last_runs:
            m = metrics_map.setdefault(sid, {})
            m.update(dict(
                last_run_status=st,
                last_run_started_at=st_at.isoformat() if st_at else None,
                last_run_ended_at=ed_at.isoformat() if ed_at else None,
                last_run_duration_ms=dur,
            ))

    def _row(site):
        m = metrics_map.get(site.id, {})
        return dict(
            id=site.id,
            name=getattr(site, "name", f"Site {site.id}") or f"Site {site.id}",
            metrics=dict(
                applied=int(m.get("applied") or 0),
                pending=int(m.get("pending") or 0),
                skipped=int(m.get("skipped") or 0),
                running=bool(m.get("running")),
                last_run_status=m.get("last_run_status"),
                last_run_started_at=m.get("last_run_started_at"),
                last_run_ended_at=m.get("last_run_ended_at"),
                last_run_duration_ms=m.get("last_run_duration_ms"),
            )
        )

    next_cursor_id = sites[-1].id if sites else None
    has_more = bool(sites) and (len(sites) == limit)

    return jsonify({
        "ok": True,
        "rows": [_row(s) for s in sites],
        "next_cursor_id": next_cursor_id,
        "has_more": has_more,
    })

# ---- 実行履歴（キーセット） ----
@admin_bp.route("/admin/internal-seo/list", methods=["GET"])
@admin_required_effective
def admin_internal_seo_list():

    site_id = request.args.get("site_id", type=int)
    status  = request.args.get("status")  # e.g. 'error', 'success', 'running', 'queued'
    limit = min(max(request.args.get("limit", type=int, default=50), 1), 200)
    cursor_ts_str = request.args.get("cursor_ts")
    cursor_id = request.args.get("cursor_id", type=int)

    q = InternalSeoRun.query.options(
        load_only(
            InternalSeoRun.id,
            InternalSeoRun.site_id,
            InternalSeoRun.status,
            InternalSeoRun.job_kind,
            InternalSeoRun.started_at,
            InternalSeoRun.ended_at,
            InternalSeoRun.duration_ms,
        ),
        defer(InternalSeoRun.stats),
    )

    if site_id:
        q = q.filter(InternalSeoRun.site_id == site_id)
    # 例: /admin/internal-seo/list?status=error で失敗ランのみ取得
    if status:
        # InternalSeoRun 側のステータスでフィルタ（'running'|'success'|'error' など）
        q = q.filter(InternalSeoRun.status == status)    

    q = q.order_by(desc(InternalSeoRun.started_at), desc(InternalSeoRun.id))

    if cursor_ts_str:
        try:
            cursor_ts = datetime.fromisoformat(cursor_ts_str.replace("Z", "+00:00"))
        except Exception:
            cursor_ts = None
        if cursor_ts is not None and cursor_id is not None:
            q = q.filter(
                or_(
                    InternalSeoRun.started_at < cursor_ts,
                    and_(InternalSeoRun.started_at == cursor_ts, InternalSeoRun.id < cursor_id),
                )
            )

    items = q.limit(limit).all()
    next_cursor_ts = items[-1].started_at.isoformat() if items else None
    next_cursor_id = items[-1].id if items else None
    has_more = bool(items) and (len(items) == limit)

    def _row(r):
        return dict(
            id=r.id,
            site_id=r.site_id,
            status=r.status,
            job_kind=r.job_kind,
            started_at=r.started_at.isoformat() if r.started_at else None,
            ended_at=r.ended_at.isoformat() if r.ended_at else None,
            duration_ms=r.duration_ms,
        )

    return jsonify(dict(
        ok=True,
        rows=[_row(r) for r in items],
        next_cursor_ts=next_cursor_ts,
        next_cursor_id=next_cursor_id,
        has_more=has_more,
    ))

# ---- 失敗ジョブの一括リトライ（error -> queued）※任意API ----
@admin_bp.route("/admin/internal-seo/retry-failed", methods=["POST"])
@admin_required_effective
def admin_internal_seo_retry_failed():
    """
    internal_seo_job_queue の status='error' を 'queued' に戻す。
    - site_id を指定すれば、そのサイトだけを対象に再投入できる。
    - running/queued のものは対象外。
    返却: {"ok": true, "requeued": n}
    """

    site_id = None
    if request.is_json:
        site_id = (request.get_json(silent=True) or {}).get("site_id")
    if site_id is None:
        # フォームからのPOSTも許容
        site_id = request.form.get("site_id", type=int)
    try:
        if site_id is not None:
            res = db.session.execute(text("""
                UPDATE internal_seo_job_queue
                   SET status='queued', updated_at=now(), message=NULL, started_at=NULL, ended_at=NULL
                 WHERE status='error' AND site_id=:sid
            """), {"sid": int(site_id)})
        else:
            res = db.session.execute(text("""
                UPDATE internal_seo_job_queue
                   SET status='queued', updated_at=now(), message=NULL, started_at=NULL, ended_at=NULL
                 WHERE status='error'
            """))
        db.session.commit()
        cnt = int(res.rowcount or 0)
        return jsonify({"ok": True, "requeued": cnt})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

# ---- 手動実行（非同期トリガ） ----
@admin_bp.route("/admin/internal-seo/run", methods=["POST"])
@admin_required_effective
def admin_internal_seo_run():

    site_id = request.form.get("site_id", type=int)
    if not site_id:
        flash("site_id は必須です", "warning")
        return redirect(url_for("admin.admin_internal_seo_index"))

    def _env_int(key: str, default: int) -> int: return int(os.getenv(key, default))
    def _env_float(key: str, default: float) -> float: return float(os.getenv(key, default))

    pages         = request.form.get("pages",         type=int,   default=_env_int("INTERNAL_SEO_PAGES", 10))
    per_page      = request.form.get("per_page",      type=int,   default=_env_int("INTERNAL_SEO_PER_PAGE", 100))
    min_score     = request.form.get("min_score",     type=float, default=_env_float("INTERNAL_SEO_MIN_SCORE", 0.05))
    max_k         = request.form.get("max_k",         type=int,   default=_env_int("INTERNAL_SEO_MAX_K", 80))
    limit_sources = request.form.get("limit_sources", type=int,   default=_env_int("INTERNAL_SEO_LIMIT_SOURCES", 200))
    limit_posts   = request.form.get("limit_posts",   type=int,   default=_env_int("INTERNAL_SEO_LIMIT_POSTS", 50))
    incremental   = request.form.get("incremental", default="true").lower() != "false"

    db.session.execute(text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        VALUES
          (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
           :incremental, 'admin-ui', 'queued', now())
    """), dict(
        site_id=site_id, pages=pages, per_page=per_page, min_score=min_score, max_k=max_k,
        limit_sources=limit_sources, limit_posts=limit_posts, incremental=incremental,
    ))
    db.session.commit()

    flash(f"Site {site_id} の内部SEOをジョブキューに登録しました。ワーカーが順次実行します。", "success")
    return redirect(url_for("admin.admin_internal_seo_index", site_id=site_id), code=303)

# ---- まとめ実行（※このUIでは個別実行を推し、APIは互換維持） ----
@admin_bp.route("/admin/internal-seo/run-batch", methods=["POST"])
@admin_required_effective
def admin_internal_seo_run_batch():

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        site_ids = payload.get("site_ids") or payload.get("site_ids[]") or []
        params = payload
    else:
        site_ids = request.form.getlist("site_ids[]") or request.form.getlist("site_ids") or []
        params = request.form
    try:
        site_ids = [int(s) for s in site_ids if str(s).strip()]
    except Exception:
        return jsonify({"ok": False, "error": "invalid site_ids"}), 400
    if not site_ids:
        return jsonify({"ok": False, "error": "site_ids required"}), 400

    def _env_int(key: str, default: int) -> int: return int(os.getenv(key, default))
    def _env_float(key: str, default: float) -> float: return float(os.getenv(key, default))

    pages         = int(params.get("pages",         _env_int("INTERNAL_SEO_PAGES", 10)))
    per_page      = int(params.get("per_page",      _env_int("INTERNAL_SEO_PER_PAGE", 100)))
    min_score     = float(params.get("min_score",   _env_float("INTERNAL_SEO_MIN_SCORE", 0.05)))
    max_k         = int(params.get("max_k",         _env_int("INTERNAL_SEO_MAX_K", 80)))
    limit_sources = int(params.get("limit_sources", _env_int("INTERNAL_SEO_LIMIT_SOURCES", 200)))
    limit_posts   = int(params.get("limit_posts",   _env_int("INTERNAL_SEO_LIMIT_POSTS", 50)))
    incremental   = str(params.get("incremental", "true")).lower() != "false"

    existing_site_ids = {s.id for s in Site.query.with_entities(Site.id).filter(Site.id.in_(site_ids)).all()}
    enqueued, skipped, errors = 0, [], []

    for sid in site_ids:
        if sid not in existing_site_ids:
            skipped.append({"site_id": sid, "reason": "site-not-found"})
            continue
        try:
            db.session.execute(text("""
                INSERT INTO internal_seo_job_queue
                  (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
                   incremental, job_kind, status, created_at)
                VALUES
                  (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
                   :incremental, 'admin-ui-batch', 'queued', now())
            """), dict(
                site_id=sid, pages=pages, per_page=per_page, min_score=min_score, max_k=max_k,
                limit_sources=limit_sources, limit_posts=limit_posts, incremental=incremental,
            ))
            enqueued += 1
        except Exception as e:
            errors.append({"site_id": sid, "error": str(e)})
    db.session.commit()

    return jsonify({"ok": True, "enqueued": enqueued, "skipped": skipped, "errors": errors})

# ---- 容量メーター ----
@admin_bp.route("/admin/internal-seo/capacity", methods=["GET"])
@admin_required_effective
def admin_internal_seo_capacity():
    

    max_parallel = int(os.getenv("INTERNAL_SEO_WORKER_PARALLELISM", 3))

    running = db.session.execute(text("SELECT COUNT(*) FROM internal_seo_runs WHERE status='running'")).scalar() or 0
    queued = db.session.execute(text("SELECT COUNT(*) FROM internal_seo_job_queue WHERE status IN ('queued','running')")).scalar() or 0

    available = max(0, max_parallel - int(running))
    suggest_batch_size = min(available, 5)

    payload = {
        "ok": True,
        "max_parallel": int(max_parallel),
        "running": int(running),
        "queued": int(queued),
        "available": int(available),
        "suggest_batch_size": int(suggest_batch_size),
    }
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

# ---- 詳細ログ（アクション） ----
@admin_bp.route("/admin/internal-seo/actions", methods=["GET"])
@admin_required_effective
def admin_internal_seo_actions():
    
    site_id = request.args.get("site_id", type=int)
    post_id = request.args.get("post_id", type=int)
    status  = request.args.get("status")
    limit   = min(max(request.args.get("limit", type=int, default=50), 1), 100)

    cursor = request.args.get("cursor")
    cursor_ts = None
    cursor_id = None
    if cursor:
        try:
            ts_str, id_str = cursor.rsplit(".", 1)
            cursor_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            cursor_id = int(id_str)
        except Exception:
            cursor_ts = None
            cursor_id = None

    q = InternalLinkAction.query.options(
        load_only(
            InternalLinkAction.id,
            InternalLinkAction.site_id,
            InternalLinkAction.post_id,
            InternalLinkAction.target_post_id,
            InternalLinkAction.anchor_text,
            InternalLinkAction.position,
            InternalLinkAction.status,
            InternalLinkAction.applied_at,
            InternalLinkAction.diff_before_excerpt,
            InternalLinkAction.diff_after_excerpt,
        )
    )

    if site_id:
        q = q.filter(InternalLinkAction.site_id == site_id)
    if post_id:
        q = q.filter(InternalLinkAction.post_id == post_id)
    if status:
        q = q.filter(InternalLinkAction.status == status)

    q = q.order_by(desc(InternalLinkAction.applied_at), desc(InternalLinkAction.id))

    if cursor_ts is not None and cursor_id is not None:
        q = q.filter(
            or_(
                InternalLinkAction.applied_at < cursor_ts,
                and_(InternalLinkAction.applied_at == cursor_ts, InternalLinkAction.id < cursor_id),
            )
        )

    rows = q.limit(limit).all()

    post_ids   = {r.post_id for r in rows if r.post_id}
    target_ids = {r.target_post_id for r in rows if r.target_post_id}
    all_ids = list(post_ids | target_ids)
    url_map = {}
    if all_ids:
        cu = (
            ContentIndex.query
            .with_entities(ContentIndex.wp_post_id, ContentIndex.url)
            .filter(ContentIndex.wp_post_id.in_(all_ids))
            .all()
        )
        url_map = {int(pid): (url or "") for (pid, url) in cu}

    def _row(r: InternalLinkAction):
        return dict(
            id=r.id,
            site_id=r.site_id,
            post_id=r.post_id,
            post_url=url_map.get(r.post_id, ""),
            target_post_id=r.target_post_id,
            target_url=url_map.get(r.target_post_id, ""),
            anchor_text=r.anchor_text,
            position=r.position,
            status=r.status,
            applied_at=r.applied_at.isoformat() if r.applied_at else None,
            diff_before_excerpt=(r.diff_before_excerpt or "")[:280],
            diff_after_excerpt=(r.diff_after_excerpt or "")[:280],
        )

    next_cursor = None
    has_more = bool(rows) and (len(rows) == limit)
    if rows:
        last = rows[-1]
        ts = last.applied_at.isoformat() if last.applied_at else "1970-01-01T00:00:00+00:00"
        next_cursor = f"{ts}.{last.id}"

    return jsonify(dict(
        ok=True,
        rows=[_row(r) for r in rows],
        next_cursor=next_cursor,
        has_more=has_more,
    ))

# ---- 全サイト一括 enqueue（まだ queued/running でないサイトのみ投入）----
@admin_bp.route("/admin/internal-seo/enqueue-all", methods=["POST"])
@admin_required_effective
def admin_internal_seo_enqueue_all():

    # 受け取りパラメータ（未指定なら .env / 環境変数 → 既定値 の順）
    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, default))
        except Exception:
            return default

    def _env_float(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, default))
        except Exception:
            return default

    # フロント（fetch）から JSON で任意の既定値を上書き可能
    params = request.get_json(silent=True) or {}
    pages         = int(params.get("pages",         _env_int("INTERNAL_SEO_PAGES", 10)))
    per_page      = int(params.get("per_page",      _env_int("INTERNAL_SEO_PER_PAGE", 100)))
    min_score     = float(params.get("min_score",   _env_float("INTERNAL_SEO_MIN_SCORE", 0.05)))
    max_k         = int(params.get("max_k",         _env_int("INTERNAL_SEO_MAX_K", 80)))
    limit_sources = int(params.get("limit_sources", _env_int("INTERNAL_SEO_LIMIT_SOURCES", 200)))
    limit_posts   = int(params.get("limit_posts",   _env_int("INTERNAL_SEO_LIMIT_POSTS", 50)))
    incremental   = str(params.get("incremental", "true")).lower() != "false"

    # すでに queued/running のサイトは除外して INSERT ... SELECT
    # ※ internal_seo_job_queue の必須カラムに合わせて構成
    sql = text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        SELECT
          s.id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
          :incremental, 'admin-bulk', 'queued', NOW()
        FROM site s
        LEFT JOIN internal_seo_job_queue q
               ON q.site_id = s.id
              AND q.status IN ('queued','running')
        WHERE q.site_id IS NULL
    """)
    res = db.session.execute(sql, dict(
        pages=pages, per_page=per_page, min_score=min_score, max_k=max_k,
        limit_sources=limit_sources, limit_posts=limit_posts, incremental=incremental,
    ))
    db.session.commit()

    inserted = res.rowcount if res.rowcount is not None else 0
    return jsonify({"ok": True, "inserted": int(inserted)})

# ---- KPI（全体サマリ：登録/キュー/実行/本日の適用/H内除去/旧仕様削除） ----
@admin_bp.route("/admin/internal-seo/kpis", methods=["GET"])
@admin_required_effective
def admin_internal_seo_kpis():
    days = int(request.args.get("days", 7))
    since = (datetime.now(JST) - timedelta(days=days)).astimezone(timezone.utc)

    # 概況（サイト数/キュー/ランニング）
    overview_sql = text("""
      SELECT
        (SELECT COUNT(*) FROM site) AS total_sites,
        (SELECT COUNT(*) FROM internal_seo_job_queue WHERE status='queued')  AS queued_sites,
        (SELECT COUNT(*) FROM internal_seo_job_queue WHERE status='running') AS running_sites
    """)
    overview = db.session.execute(overview_sql).mappings().first() or {}

    # 適用・除去の集計（期間合計 + 本日）
    agg_sql = text("""
      WITH logs AS (
        SELECT
          l.id,
          l.status,
          COALESCE(l.applied_links, (l.details->>'applied_links')::int) AS applied_links,
          COALESCE(l.removed_in_headings, (l.details->>'removed_in_headings')::int) AS removed_in_headings,
          COALESCE(l.legacy_removed, (l.details->>'legacy_removed')::int) AS legacy_removed,
          l.created_at AT TIME ZONE 'UTC' AS created_utc
        FROM internal_seo_job_log l
        WHERE l.created_at >= :since
      )
      SELECT
        (SELECT COALESCE(SUM(applied_links),0)
           FROM logs
          WHERE status='applied'
            AND created_utc::date = (now() AT TIME ZONE 'UTC')::date) AS applied_today,
        (SELECT COALESCE(SUM(removed_in_headings),0) FROM logs) AS removed_in_h_total,
        (SELECT COALESCE(SUM(legacy_removed),0)       FROM logs) AS legacy_removed_total
    """)
    agg = db.session.execute(agg_sql, {"since": since}).mappings().first() or {}

    payload = {
        "total_sites": int(overview.get("total_sites") or 0),
        "queued_sites": int(overview.get("queued_sites") or 0),
        "running_sites": int(overview.get("running_sites") or 0),
        "applied_today": int(agg.get("applied_today") or 0),
        "removed_in_h_total": int(agg.get("removed_in_h_total") or 0),
        "legacy_removed_total": int(agg.get("legacy_removed_total") or 0),
        "days": days,
    }
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

# ---- リアルタイムログ（増分） ----
@admin_bp.route("/admin/internal-seo/logs", methods=["GET"])
@admin_required_effective
def admin_internal_seo_logs():
    """
    クエリ:
      - limit: 取得件数（デフォルト 50, 最大 200）
      - since: ISO8601（JST/UTC可）これ以降のログを返す
    返却: id 降順（新→古）。フロントは上に積む。
    """
    limit = min(int(request.args.get("limit", 50)), 200)
    since_str = request.args.get("since")
    since_dt = None
    if since_str:
        try:
            s = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
            since_dt = s.astimezone(timezone.utc)
        except Exception:
            since_dt = None

    params = {"limit": limit}
    where_add = ""
    if since_dt:
        where_add = "AND l.created_at > :since"
        params["since"] = since_dt

    sql = text(f"""
      SELECT
        l.id,
        l.site_id,
        s.name     AS site_name,
        u.username AS username,
        l.status,
        COALESCE(l.reason, (l.details->>'reason')) AS reason,
        COALESCE(l.applied_links, (l.details->>'applied_links')::int) AS applied_links,
        COALESCE(l.removed_in_headings, (l.details->>'removed_in_headings')::int) AS removed_in_headings,
        COALESCE(l.legacy_removed, (l.details->>'legacy_removed')::int) AS legacy_removed,
        l.created_at
      FROM internal_seo_job_log l
      LEFT JOIN site  s ON s.id = l.site_id
      LEFT JOIN "user" u ON u.id = s.user_id
      WHERE 1=1
        {where_add}
      ORDER BY l.id DESC
      LIMIT :limit
    """)
    rows = db.session.execute(sql, params).mappings().all() or []

    out = []
    for r in rows:
        out.append({
            "id": int(r.get("id")),
            "site_id": r.get("site_id"),
            "site_name": r.get("site_name"),
            "username": r.get("username"),
            "status": r.get("status"),
            "reason": r.get("reason"),
            "applied_links": int(r.get("applied_links") or 0),
            "removed_in_headings": int(r.get("removed_in_headings") or 0),
            "legacy_removed": int(r.get("legacy_removed") or 0),
            "created_at": r.get("created_at").astimezone(JST).isoformat(timespec="seconds"),
        })
    return jsonify({"logs": out})


# ---------------------------------------------------------------------------
# 🆕 管理UI: ユーザー別 内部SEOスケジュール 導線 & API
#    パスは /admin/iseo/schedules/... に統一（既存の /admin/internal-seo/* と分離）
# ---------------------------------------------------------------------------

@admin_bp.route("/admin/iseo/schedules")
@admin_required_effective
def admin_iseo_user_schedules_page():
    """
    一覧ページ（テンプレートは別PRで用意）
    """
    return render_template("admin/iseo_user_schedules.html")  # テンプレが無ければ一旦 500 でもOK


def _get_or_create_user_schedule(uid: int) -> InternalSeoUserSchedule:
    sch = InternalSeoUserSchedule.query.filter_by(user_id=uid).one_or_none()
    if not sch:
        sch = InternalSeoUserSchedule(user_id=uid)
        from app import db
        db.session.add(sch)
        db.session.commit()
    return sch


@admin_bp.route("/admin/iseo/schedules/status")
@admin_required_effective
def admin_iseo_user_schedules_status():
    """
    一覧テーブル用のJSON。
    返却: pending件数 / 状態 / 「開始済み」判定 / 直近実行の所要 / 直近7日スループット / 予測消化日数 /
         累計 applied/processed・平均リンク数・last_error。
    ※ 互換のため applied_24h / processed_24h も当面返す（同値または別集計）。
    """
    from app import db
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func, text
    from app.models import User, Site, InternalSeoUserSchedule, InternalSeoUserRun
    now_utc = datetime.now(timezone.utc)
    since_7d = now_utc - timedelta(days=7)
    # 対象ユーザーは「サイトを1つ以上持つユーザー」を基本にする
    user_rows = (
        db.session.query(
            User.id.label("user_id"),
            User.username,
            func.count(Site.id).label("site_cnt"),
        )
        .outerjoin(Site, Site.user_id == User.id)
        .group_by(User.id, User.username)
        .having(func.count(Site.id) > 0)
        .all()
    )

    # スケジュール/直近ランの付帯情報
    result = []
    for u in user_rows:
        sch = InternalSeoUserSchedule.query.filter_by(user_id=u.user_id).one_or_none()
        last_run = (
            InternalSeoUserRun.query
            .filter_by(user_id=u.user_id)
            .order_by(InternalSeoUserRun.started_at.desc(), InternalSeoUserRun.id.desc())
            .first()
        )
        # 「開始済み」判定（is_enabled かつ 1度でも実行したことがある）
        is_started = bool(getattr(sch, "is_enabled", False) and (
            (last_run is not None) or getattr(sch, "last_run_at", None)
        ))
        # pending 件数（ユーザー配下サイトの pending の distinct post_id）
        pending_cnt = db.session.execute(
            text("""
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT a.post_id
                    FROM internal_link_actions a
                    JOIN site s ON s.id = a.site_id
                   WHERE s.user_id = :uid
                     AND a.status = 'pending'
                ) t
            """),
            {"uid": u.user_id}
        ).scalar() or 0
        # ✅ 全期間の累計（時間条件を外す）
        agg_all = (
            db.session.query(
                func.coalesce(func.sum(InternalSeoUserRun.applied), 0),
                func.coalesce(func.sum(InternalSeoUserRun.processed_posts), 0),
            )
            .filter(InternalSeoUserRun.user_id == u.user_id)
            .one()
        )
        applied_total = int(agg_all[0] or 0)
        processed_total = int(agg_all[1] or 0)

        # 互換：従来の24hキーは当面返す（必要に応じて後で削除）
        applied_24h = applied_total
        processed_24h = processed_total
        # 直近7日の処理記事スループット（実績ベース）
        agg_7d = (
            db.session.query(
                func.coalesce(func.sum(InternalSeoUserRun.processed_posts), 0)
            )
            .filter(
                InternalSeoUserRun.user_id == u.user_id,
                InternalSeoUserRun.started_at >= since_7d
            )
            .one()
        )
        throughput_7d = int(agg_7d[0] or 0)  # 7日間で処理した記事数
        avg_per_day = float(throughput_7d) / 7.0 if throughput_7d else 0.0
        # 予測消化日数：pending を 7日平均/日の処理数で割る（0なら None）
        pred_days = (float(pending_cnt) / avg_per_day) if avg_per_day > 0 else None

        # 直近実行の所要（分）
        if last_run and getattr(last_run, "started_at", None) and getattr(last_run, "finished_at", None):
            dur_sec = (last_run.finished_at - last_run.started_at).total_seconds()
            duration_min = round(dur_sec / 60.0, 1)
        else:
            duration_min = None

        # 平均リンク数/記事（累計）
        avg_links_per_post = (float(applied_total) / float(processed_total)) if processed_total > 0 else 0.0

        result.append({
            "user_id": u.user_id,
            "username": u.username,
            "sites": int(u.site_cnt or 0),
            "is_enabled": bool(getattr(sch, "is_enabled", False)),
            "is_started": is_started,
            "status": getattr(sch, "status", "idle") if sch else "idle",
            "last_run_at": getattr(sch, "last_run_at", None).isoformat() if sch and sch.last_run_at else None,
            "next_run_at": getattr(sch, "next_run_at", None).isoformat() if sch and sch.next_run_at else None,
            "tick_interval_sec": getattr(sch, "tick_interval_sec", None) if sch else None,
            "budget_per_tick": getattr(sch, "budget_per_tick", None) if sch else None,
            "rate_limit_per_min": getattr(sch, "rate_limit_per_min", None) if sch else None,
            "last_error": getattr(sch, "last_error", None) if sch else None,
            "pending": int(pending_cnt),
            # 直近7日スループットと予測
            "throughput_7d": throughput_7d,
            "avg_per_day": avg_per_day,
            "pred_days": pred_days,
            # 新：全期間累計
            "applied_total": applied_total,
            "processed_total": processed_total,
            "avg_links_per_post": avg_links_per_post,
            # 旧：後方互換（テンプレ移行中は残す）
            "applied_24h": applied_24h,
            "processed_24h": processed_24h,
            "last_result": {
                "status": getattr(last_run, "status", None) if last_run else None,
                "applied": getattr(last_run, "applied", None) if last_run else None,
                "processed_posts": getattr(last_run, "processed_posts", None) if last_run else None,
                "finished_at": getattr(last_run, "finished_at", None).isoformat() if last_run and last_run.finished_at else None,
                "duration_min": duration_min,
            },
        })
    return jsonify({"items": result})


def _parse_user_ids_from_request():
    data = request.get_json(silent=True) or {}
    ids = data.get("user_ids") or data.get("ids") or []
    # フォームPOST対応
    if not ids and "user_ids" in request.form:
        ids = request.form.getlist("user_ids")
    try:
        return [int(x) for x in ids]
    except Exception:
        return []


@admin_bp.route("/admin/iseo/schedules/bulk_enable", methods=["POST"])
@admin_required_effective
def admin_iseo_user_schedules_bulk_enable():
    """
    is_enabled=True, status=queued にして即時1tickを投入
    """
    from app import db
    from flask import current_app
    from app.services.internal_seo.user_scheduler import run_user_tick
    app = current_app._get_current_object()
    ids = _parse_user_ids_from_request()
    if not ids:
        return jsonify({"ok": False, "error": "user_ids required"}), 400
    for uid in ids:
        sch = _get_or_create_user_schedule(uid)
        sch.is_enabled = True
        sch.status = "queued"
        db.session.add(sch)
    db.session.commit()
    # 即時に1回だけ同期 tick（軽量・安全）
    for uid in ids:
        try:
            run_user_tick(app, uid, force=True)
        except Exception:
            current_app.logger.exception("[iseo] run_user_tick (bulk_enable) failed user_id=%s", uid)
    return jsonify({"ok": True, "enabled": ids})


@admin_bp.route("/admin/iseo/schedules/bulk_disable", methods=["POST"])
@admin_required_effective
def admin_iseo_user_schedules_bulk_disable():
    """
    完全停止：is_enabled=False, status=idle, next_run_at=NULL
    """
    from app import db
    ids = _parse_user_ids_from_request()
    if not ids:
        return jsonify({"ok": False, "error": "user_ids required"}), 400
    q = InternalSeoUserSchedule.query.filter(InternalSeoUserSchedule.user_id.in_(ids))
    for sch in q.all():
        sch.is_enabled = False
        sch.status = "idle"
        sch.next_run_at = None
        db.session.add(sch)
    db.session.commit()
    return jsonify({"ok": True, "disabled": ids})


@admin_bp.route("/admin/iseo/schedules/bulk_pause", methods=["POST"])
@admin_required_effective
def admin_iseo_user_schedules_bulk_pause():
    """
    一時停止：status=paused（is_enabledは保持）
    """
    from app import db
    ids = _parse_user_ids_from_request()
    if not ids:
        return jsonify({"ok": False, "error": "user_ids required"}), 400
    q = InternalSeoUserSchedule.query.filter(InternalSeoUserSchedule.user_id.in_(ids))
    for sch in q.all():
        sch.status = "paused"
        db.session.add(sch)
    db.session.commit()
    return jsonify({"ok": True, "paused": ids})


@admin_bp.route("/admin/iseo/schedules/bulk_resume", methods=["POST"])
@admin_required_effective
def admin_iseo_user_schedules_bulk_resume():
    """
    再開：status=queued に戻し、即時 tick を投入
    """
    from app import db
    from flask import current_app
    from app.services.internal_seo.user_scheduler import run_user_tick
    app = current_app._get_current_object()
    ids = _parse_user_ids_from_request()
    if not ids:
        return jsonify({"ok": False, "error": "user_ids required"}), 400
    q = InternalSeoUserSchedule.query.filter(InternalSeoUserSchedule.user_id.in_(ids))
    for sch in q.all():
        sch.status = "queued"
        db.session.add(sch)
    db.session.commit()
    # 即時に1回だけ同期 tick
    for uid in ids:
        try:
            run_user_tick(app, uid, force=True)
        except Exception:
            current_app.logger.exception("[iseo] run_user_tick (bulk_resume) failed user_id=%s", uid)
    return jsonify({"ok": True, "resumed": ids})


@admin_bp.route("/admin/iseo/schedules/run_once", methods=["POST"])
@admin_required_effective
def admin_iseo_user_schedules_run_once():
    """
    即時に 1 tick 実行（is_enabled 無視で単発）
    """
    from flask import current_app
    from app.services.internal_seo.user_scheduler import run_user_tick
    app = current_app._get_current_object()
    uid = (request.get_json(silent=True) or {}).get("user_id") or request.form.get("user_id")
    try:
        uid = int(uid)
    except Exception:
        return jsonify({"ok": False, "error": "user_id required"}), 400
    # 同期実行（安全・即時）
    try:
        res = run_user_tick(app, uid, force=True)
        # 実行直後の最新メトリクスを再集計して返す（UI即時反映用）
        from app import db
        from sqlalchemy import func, text
        from app.models import InternalSeoUserRun, InternalSeoUserSchedule
        from datetime import datetime, timedelta, timezone
        now_utc = datetime.now(timezone.utc)
        since_24h = now_utc - timedelta(hours=24)

        pending_cnt = db.session.execute(
            text("""
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT a.post_id
                    FROM internal_link_actions a
                    JOIN site s ON s.id = a.site_id
                   WHERE s.user_id = :uid
                     AND a.status = 'pending'
                ) t
            """),
            {"uid": uid}
        ).scalar() or 0

        agg_24h = (
            db.session.query(
                func.coalesce(func.sum(InternalSeoUserRun.applied), 0),
                func.coalesce(func.sum(InternalSeoUserRun.processed_posts), 0),
            )
            .filter(
                InternalSeoUserRun.user_id == uid,
                InternalSeoUserRun.started_at >= since_24h
            )
            .one()
        )
        applied_24h = int(agg_24h[0] or 0)
        processed_24h = int(agg_24h[1] or 0)

        sch = InternalSeoUserSchedule.query.filter_by(user_id=uid).one_or_none()
        last_error = getattr(sch, "last_error", None) if sch else None

        return jsonify({
            "ok": bool(res.get("ok", False)),
            "result": res,
            "pending": int(pending_cnt),
            "applied_24h": applied_24h,
            "processed_24h": processed_24h,
            "last_error": last_error,
        })
    except Exception as e:
        current_app.logger.exception("[iseo] run_user_tick failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@admin_bp.route("/admin/iseo/schedules/<int:user_id>/runs")
@admin_required_effective
def admin_iseo_user_runs(user_id: int):
    """
    直近のユーザー実行履歴（軽量JSON）
    """
    from app.models import InternalSeoUserRun
    from sqlalchemy import func
    from datetime import datetime, timedelta, timezone
    now_utc = datetime.now(timezone.utc)
    since_24h = now_utc - timedelta(hours=24)
    q = (
        InternalSeoUserRun.query
        .filter_by(user_id=user_id)
        .order_by(InternalSeoUserRun.started_at.desc(), InternalSeoUserRun.id.desc())
        .limit(50)
    )
    items = [{
        "id": r.id,
        "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "applied": r.applied,
        "swapped": r.swapped,
        "skipped": r.skipped,
        "processed_posts": r.processed_posts,
    } for r in q.all()]
    # 直近24hの合計も一緒に返す
    agg = (
        InternalSeoUserRun.query
        .with_entities(
            func.coalesce(func.sum(InternalSeoUserRun.applied), 0),
            func.coalesce(func.sum(InternalSeoUserRun.processed_posts), 0)
        )
        .filter(
            InternalSeoUserRun.user_id == user_id,
            InternalSeoUserRun.started_at >= since_24h
        )
        .one()
    )
    return jsonify({
        "user_id": user_id,
        "items": items,
        "applied_24h": int(agg[0] or 0),
        "processed_24h": int(agg[1] or 0),
    })

# ---- 🆕 適用済み記事一覧（ページ本体） ----
@admin_bp.route("/admin/iseo/applied_all", methods=["GET"])
@admin_required_effective
def admin_iseo_applied_all_page():
    # 一覧ページは user_id を受け取り、テンプレ側で持ち回す
    # ※ 未指定でも描画自体は行う（データAPI側で必須化）
    user_id = request.args.get("user_id", type=int)
    return render_template("admin/iseo_applied_all.html", current_user_id=user_id)

# ---- 🆕 適用済み記事一覧：集計データ（記事×version） ----
@admin_bp.route("/admin/iseo/applied_all/data", methods=["GET"])
@admin_required_effective
def admin_iseo_applied_all_data():
    # フィルタ
    user_id = request.args.get("user_id", type=int)
    site_id = request.args.get("site_id", type=int)
    version = request.args.get("version", type=int)  # ← 最新Versionでの絞り込み
    date_from = request.args.get("date_from")
    date_to   = request.args.get("date_to")
    limit = min(max(request.args.get("limit", default=50, type=int), 1), 200)

    # user_id は必須。未指定なら 400
    if user_id is None:
        return jsonify({"ok": False, "error": "user_id required"}), 400

    # カーソル（last_applied_at, latest_ver の複合）
    cursor = request.args.get("cursor")  # "ISO8601.VER"
    cur_ts = None
    cur_ver = None
    if cursor:
        try:
            ts_s, ver_s = cursor.rsplit(".", 1)
            cur_ts = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
            cur_ver = int(ver_s)
        except Exception:
            cur_ts = None
            cur_ver = None

    params = {}
    params["user_id"] = user_id
    # 動的条件（この段階では CTE 後の最終SELECTに適用）
    where_final = ["u.id = :user_id"]
    if site_id is not None:
        where_final.append("s.id = :site_id")
        params["site_id"] = site_id
    if version is not None:
        where_final.append("l.latest_ver = :version")
        params["version"] = version
    if date_from:
        where_final.append("l.last_applied_at >= :date_from")
        params["date_from"] = datetime.fromisoformat(f"{date_from}T00:00:00+00:00")
    if date_to:
        where_final.append("l.last_applied_at < :date_to")
        params["date_to"] = datetime.fromisoformat(f"{date_to}T23:59:59.999999+00:00")
    # カーソル: (last_applied_at desc, latest_ver desc) の keyset
    if cur_ts is not None and cur_ver is not None:
        where_final.append("(l.last_applied_at < :cur_ts OR (l.last_applied_at = :cur_ts AND l.latest_ver < :cur_ver))")
        params["cur_ts"] = cur_ts
        params["cur_ver"] = cur_ver
    where_sql = " AND ".join(where_final) if where_final else "1=1"

    sql = text(f"""
      WITH latest AS (
        SELECT
          s.user_id,
          ila.site_id,
          ila.post_id,
          MAX(ila.link_version) AS latest_ver,
          MAX(ila.applied_at) FILTER (WHERE ila.status = 'applied') AS last_applied_at
        FROM internal_link_actions ila
        JOIN site s ON s.id = ila.site_id
        WHERE s.user_id = :user_id
        GROUP BY s.user_id, ila.site_id, ila.post_id
      )
      SELECT
        u.id   AS user_id,
        COALESCE(u.username, (u.last_name || u.first_name)) AS user_name,
        s.id   AS site_id,
        s.name AS site_name,
        ci.wp_post_id AS post_id,
        ci.title      AS src_title,
        ci.url        AS src_url,
        l.latest_ver  AS link_version,
        COUNT(a.*)    AS candidate_count,
        SUM(CASE WHEN a.status = 'applied' THEN 1 ELSE 0 END) AS applied_count,
        l.last_applied_at AS last_applied_at
      FROM latest l
      JOIN site s   ON s.id = l.site_id
      JOIN "user" u ON u.id = s.user_id
      LEFT JOIN content_index ci
        ON ci.site_id = l.site_id
       AND ci.wp_post_id = l.post_id
      LEFT JOIN internal_link_actions a
        ON a.site_id = l.site_id
       AND a.post_id = l.post_id
       AND a.link_version = l.latest_ver
      WHERE {where_sql}
      GROUP BY u.id, user_name, s.id, s.name, ci.wp_post_id, ci.title, ci.url, l.latest_ver, l.last_applied_at
      ORDER BY l.last_applied_at DESC NULLS LAST, l.latest_ver DESC
      LIMIT :limit
    """)
    params["limit"] = limit

    rows = db.session.execute(sql, params).mappings().all() or []

    def _row(r):
        return dict(
            user_id=r.get("user_id"),
            user_name=r.get("user_name"),
            site_id=r.get("site_id"),
            site_name=r.get("site_name"),
            post_id=r.get("post_id"),
            src_title=r.get("src_title"),
            src_url=r.get("src_url"),
            link_version=int(r.get("link_version") or 0),   # 最新Version
            candidate_count=int(r.get("candidate_count") or 0),
            applied_count=int(r.get("applied_count") or 0),
            last_applied_at=(r.get("last_applied_at").isoformat() if r.get("last_applied_at") else None),
        )

    out_rows = [_row(r) for r in rows]
    # 次カーソル
    next_cursor = None
    has_more = len(out_rows) == limit
    if has_more:
        last = out_rows[-1]
        next_cursor = f"{last['last_applied_at']}.{last['link_version']}"

    # サイトごとにグループ化（ドロップ開閉式用）
    site_grouped = {}
    for r in out_rows:
        sid = r["site_id"]
        if sid not in site_grouped:
            site_grouped[sid] = {
                "site_id": sid,
                "site_name": r["site_name"],
                "articles": []
            }
        site_grouped[sid]["articles"].append(r)

    site_list = list(site_grouped.values())

    return jsonify({
        "ok": True,
        "rows": out_rows,
        "sites": site_list,
        "next_cursor": next_cursor,
        "has_more": has_more
    })


# ---- 🆕 明細：記事×version のリンク一覧 ----
@admin_bp.route("/admin/iseo/applied_details", methods=["GET"])
@admin_required_effective
def admin_iseo_applied_details():
    site_id = request.args.get("site_id", type=int)
    post_id = request.args.get("post_id", type=int)
    version = request.args.get("version", type=int)
    if not (site_id and post_id and version is not None):
        return jsonify({"ok": False, "error": "site_id, post_id, version required"}), 400

    q = (
        InternalLinkAction.query
        .with_entities(
            InternalLinkAction.target_post_id,
            InternalLinkAction.anchor_text,
            InternalLinkAction.position,
            InternalLinkAction.applied_at,
            InternalLinkAction.status,
        )
        .filter_by(site_id=site_id, post_id=post_id, link_version=version)
        .filter(InternalLinkAction.status == "applied")
        .order_by(InternalLinkAction.applied_at.desc(), InternalLinkAction.id.desc())
    )
    rows = q.limit(500).all()

    # target_url 解決
    tgt_ids = [r[0] for r in rows if r[0]]
    url_map = {}
    if tgt_ids:
        for pid, url in (
            db.session.query(ContentIndex.wp_post_id, ContentIndex.url)
            .filter(ContentIndex.wp_post_id.in_(tgt_ids))
            .all()
        ):
            url_map[int(pid)] = url or ""

    items = []
    for tpid, atext, pos, ap_at, st in rows:
        items.append({
            "target_url": url_map.get(tpid, ""),
            "anchor_text": atext or "",
            "position": pos or "",
            "applied_at": ap_at.isoformat() if ap_at else None,
            "status": st or "",
        })
    return jsonify({"ok": True, "items": items})


# ────────────── キーワード ──────────────

@bp.route("/<username>/keywords", methods=["GET", "POST"])
@login_required
def keywords(username):
    if current_user.username != username:
        abort(403)

    form = KeywordForm()

    user_sites = Site.query.filter_by(user_id=current_user.id).all()
    form.site_id.choices = [(0, "―― サイトを選択 ――")] + [(s.id, s.name) for s in user_sites]

    if form.validate_on_submit():
        site_id = form.site_id.data
        if site_id == 0:
            flash("サイトを選択してください。", "danger")
            return redirect(url_for("main.keywords", username=username))

        lines = [line.strip() for line in form.keywords.data.splitlines() if line.strip()]
        for word in lines:
            keyword = Keyword(
                keyword=word,
                user_id=current_user.id,
                site_id=site_id
            )
            db.session.add(keyword)
        db.session.commit()
        flash(f"{len(lines)} 件のキーワードを追加しました", "success")
        return redirect(url_for("main.keywords", username=username, site_id=site_id))

    selected_site_id = request.args.get("site_id", type=int)
    status_filter = request.args.get("status")
    selected_site = Site.query.get(selected_site_id) if selected_site_id else None

    base_query = Keyword.query.filter_by(user_id=current_user.id)
    if selected_site_id:
        base_query = base_query.filter_by(site_id=selected_site_id)
    if status_filter == "used":
        base_query = base_query.filter_by(used=True)
    elif status_filter == "unused":
        base_query = base_query.filter_by(used=False)

    all_keywords = base_query.order_by(Keyword.site_id, Keyword.id.desc()).all()
    site_map = {s.id: s.name for s in user_sites}
    grouped_keywords = defaultdict(lambda: {"site_name": "", "keywords": [], "status_filter": status_filter})

    for kw in all_keywords:
        grouped_keywords[kw.site_id]["site_name"] = site_map.get(kw.site_id, "未設定")
        grouped_keywords[kw.site_id]["keywords"].append(kw)

    return render_template(
        "keywords.html",
        form=form,
        selected_site=selected_site,
        sites=user_sites,
        grouped_keywords=grouped_keywords,
        site_map=site_map
    )


@bp.route("/api/keywords/<int:site_id>")
@login_required
def api_unused_keywords(site_id):
    offset = request.args.get("offset", 0, type=int)
    limit = 40
    keywords = Keyword.query.filter_by(user_id=current_user.id, site_id=site_id, used=False)\
        .order_by(Keyword.id.desc()).offset(offset).limit(limit).all()
    return jsonify([{"id": k.id, "keyword": k.keyword, "used": k.used} for k in keywords])


@bp.route("/api/keywords/all/<int:site_id>")
@login_required
def api_all_keywords(site_id):
    keywords = Keyword.query.filter_by(user_id=current_user.id, site_id=site_id)\
        .order_by(Keyword.id.desc()).limit(1000).all()
    return jsonify([{"id": k.id, "keyword": k.keyword} for k in keywords])


@bp.route("/<username>/keywords/edit/<int:keyword_id>", methods=["GET", "POST"])
@login_required
def edit_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)

    form = EditKeywordForm(obj=keyword)
    form.site_id.choices = [(s.id, s.name) for s in Site.query.filter_by(user_id=current_user.id).all()]

    if form.validate_on_submit():
        keyword.keyword = form.keyword.data.strip()
        keyword.site_id = form.site_id.data
        db.session.commit()
        flash("キーワードを更新しました", "success")
        return redirect(url_for("main.keywords", username=username))

    return render_template("edit_keyword.html", form=form)


@bp.route("/<username>/keywords/view/<int:keyword_id>")
@login_required
def view_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)
    return render_template("view_keyword.html", keyword=keyword)


@bp.route("/<username>/keywords/delete/<int:keyword_id>")
@login_required
def delete_keyword(username, keyword_id):
    if current_user.username != username:
        abort(403)
    keyword = Keyword.query.get_or_404(keyword_id)
    if keyword.user_id != current_user.id:
        abort(403)
    db.session.delete(keyword)
    db.session.commit()
    flash("キーワードを削除しました。", "success")
    return redirect(url_for("main.keywords", username=username))


@bp.post("/<username>/keywords/bulk-action")
@login_required
def bulk_action_keywords(username):
    if current_user.username != username:
        abort(403)
    action = request.form.get("action")
    keyword_ids = request.form.getlist("keyword_ids")

    if not keyword_ids:
        flash("対象のキーワードが選択されていません。", "warning")
        return redirect(request.referrer or url_for("main.keywords", username=username))

    if action == "delete":
        Keyword.query.filter(
            Keyword.id.in_(keyword_ids),
            Keyword.user_id == current_user.id
        ).delete(synchronize_session=False)
        db.session.commit()
        flash("選択されたキーワードを削除しました。", "success")

    return redirect(request.referrer or url_for("main.keywords", username=username))

# ────────────── chatgpt ──────────────

@bp.route("/<username>/chatgpt")
@login_required
def chatgpt(username):
    if current_user.username != username:
        abort(403)
    return render_template("chatgpt.html")




# ─────────── 認証
@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        identifier = form.identifier.data
        password = form.password.data

        user = User.query.filter(
            (User.email == identifier) | (User.username == identifier)
        ).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("ログイン成功！", "success")
            return redirect(url_for("main.dashboard", username=user.username))
        else:
            flash("ログイン情報が正しくありません。", "danger")

    return render_template("login.html", form=form)

# 既存 import に追加
from flask import render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.security import generate_password_hash
import secrets, time, unicodedata
from app.models import User
from app import db
from sqlalchemy import func
from app.forms import RealNameEmailResetRequestForm, PasswordResetSimpleForm


def _norm_name(s: str) -> str:
    # 全角/半角のゆらぎ吸収 + 空白除去（半角/全角スペース両方）
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s.replace(" ", "").replace("\u3000", "")  # 半角/全角スペース除去

@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password_username_only():
    form = RealNameEmailResetRequestForm()
    if form.validate_on_submit():
        ln = form.last_name.data.strip()
        fn = form.first_name.data.strip()
        email = form.email.data.strip().lower()

        # メール一致のユーザーを取得（メールは小文字比較）
        user = User.query.filter(func.lower(User.email) == email).first()

        # 本名一致をサーバ側で厳密チェック（表記ゆれを軽減）
        if user and _norm_name(user.last_name) == _norm_name(ln) and _norm_name(user.first_name) == _norm_name(fn):
            grant = secrets.token_urlsafe(16)
            session["pw_reset_grant"] = {"uid": user.id, "grant": grant, "ts": time.time()}
            return redirect(url_for("main.reset_password_username_only", grant=grant))

        flash("本名とメールアドレスの組み合わせが確認できませんでした。", "danger")
        return render_template("forgot_username_only.html", form=form), 400

    return render_template("forgot_username_only.html", form=form)


# ---- Step2: 新パスワード設定
@bp.route("/reset-password-simple", methods=["GET", "POST"])
def reset_password_username_only():
    # TTL（秒）…未設定なら10分
    ttl = int(current_app.config.get("USERNAME_ONLY_RESET_TTL", 600))
    grant = request.args.get("grant") or request.form.get("grant")

    data = session.get("pw_reset_grant")
    if not data or data.get("grant") != grant or (time.time() - data.get("ts", 0)) > ttl:
        flash("操作が無効または期限切れです。最初からやり直してください。", "danger")
        session.pop("pw_reset_grant", None)
        return redirect(url_for("main.forgot_password_username_only"))

    user = User.query.get(data["uid"])
    if not user:
        flash("ユーザーが見つかりません。", "danger")
        session.pop("pw_reset_grant", None)
        return redirect(url_for("main.forgot_password_username_only"))

    form = PasswordResetSimpleForm()
    # hidden に grant を入れる
    if request.method == "GET":
        form.grant.data = grant

    if form.validate_on_submit():
        # ここまで来ていれば EqualTo で一致検証済み
        new_pw = form.password.data
        user.password = generate_password_hash(new_pw, method="pbkdf2:sha256", salt_length=16)
        db.session.commit()
        session.pop("pw_reset_grant", None)
        flash("パスワードを更新しました。新しいパスワードでログインしてください。", "success")
        return redirect(url_for("main.login"))

    return render_template("reset_username_only.html",
                           form=form, username=user.username, grant=grant)


@bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # 登録用パスワードが正しいか
        if form.register_key.data != "tcctool":
            flash("登録専用パスワードが間違っています。", "danger")
            return render_template("register.html", form=form)

        # メールアドレスの重複チェック
        if User.query.filter_by(email=form.email.data).first():
            flash("このメールアドレスは既に登録されています。", "danger")
            return render_template("register.html", form=form)

        # ユーザー名（username）の重複チェック
        if User.query.filter_by(username=form.username.data).first():
            flash("このユーザー名はすでに使われています。", "danger")
            return render_template("register.html", form=form)

        # ユーザー作成・保存
        new_user = User(
            email=form.email.data,
            password=generate_password_hash(form.password.data),
            username=form.username.data,
            user_type=form.user_type.data,
            company_name=form.company_name.data,
            company_kana=form.company_kana.data,
            last_name=form.last_name.data,
            first_name=form.first_name.data,
            last_kana=form.last_kana.data,
            first_kana=form.first_kana.data,
            postal_code=form.postal_code.data,
            address=form.address.data,
            phone=form.phone.data
        )
        db.session.add(new_user)
        db.session.commit()

        flash("登録が完了しました。ログインしてください。", "success")
        return redirect(url_for(".login"))

    return render_template("register.html", form=form)



@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for(".login"))

# ────────────── プロフィール ──────────────

@bp.route("/<username>/profile", methods=["GET", "POST"])
@login_required
def profile(username):
    if current_user.username != username:
        abort(403)

    form = ProfileForm(obj=current_user)

    if form.validate_on_submit():
        # ユーザー名が仮名（user123）のままで、変更された場合のみ許可
        if current_user.username.startswith("user") and form.username.data != current_user.username:
            # 重複チェック
            if User.query.filter_by(username=form.username.data).first():
                flash("このユーザー名はすでに使われています。", "danger")
                return render_template("profile.html", form=form)
            else:
                current_user.username = form.username.data

        # 基本情報の更新
        current_user.last_name  = form.last_name.data
        current_user.first_name = form.first_name.data
        current_user.last_kana  = form.last_kana.data
        current_user.first_kana = form.first_kana.data
        current_user.phone      = form.phone.data
        current_user.postal_code = form.postal_code.data
        current_user.address    = form.address.data  # ← 統合された住所フィールドに対応

        db.session.commit()
        flash("プロフィールを更新しました。", "success")

        return redirect(url_for("main.profile", username=current_user.username))

    return render_template("profile.html", form=form)


# ────────────── ツール本体コード ──────────────

@bp.route("/")
@login_required
def root_redirect():
    return redirect(url_for("main.dashboard", username=current_user.username))


# ─────────── Dashboard
from app.models import UserSiteQuota, Article, SiteQuotaLog, Site, User, GSCDailyTotal  # ← User を追加
from app.utils.cache import cache_get_json, cache_set_json
from sqlalchemy import case, func  # ← func を追加
from flask import g
from collections import defaultdict
from datetime import datetime, timedelta, timezone  # ← JST計算のため

@bp.route("/<username>/dashboard")
@login_required
def dashboard(username):
    if current_user.username != username:
        abort(403)

    user = current_user

    # 🔸 記事統計（SQL1）
    article_stats = db.session.query(
        func.count(Article.id),
        func.sum(case((Article.status == "done", 1), else_=0)),
        func.sum(case((Article.status == "posted", 1), else_=0)),
        func.sum(case((Article.status == "error", 1), else_=0)),
        func.sum(case((Article.status.in_(["pending", "gen"]), 1), else_=0)),
    ).filter(Article.user_id == user.id).first()

    g.total_articles = article_stats[0]
    g.done = article_stats[1]
    g.posted = article_stats[2]
    g.error = article_stats[3]
    g.generating = article_stats[4]

    # 🔸 プラン別クォータ取得（SQL2）
    quotas = UserSiteQuota.query.filter_by(user_id=user.id).all()

    # 🔸 サイト使用状況を一括取得（SQL3）
    site_counts = db.session.query(
        Site.plan_type,
        func.count(Site.id)
    ).filter(Site.user_id == user.id).group_by(Site.plan_type).all()
    site_count_map = dict(site_counts)

    # 🔸 ログを一括取得（SQL4）
    # 🔸 ログを軽量取得（各プラン最大10件まで）
    log_map = defaultdict(list)
    for plan in set([q.plan_type for q in quotas]):
        logs = SiteQuotaLog.query.filter_by(user_id=user.id, plan_type=plan) \
            .order_by(SiteQuotaLog.created_at.desc()) \
            .limit(10).all()
        log_map[plan] = logs


    # 🔸 プラン別構成
    plans = {}
    for q in quotas:
        used = site_count_map.get(q.plan_type, 0)
        total = q.total_quota or 0
        remaining = max(total - used, 0)
        plans[q.plan_type] = {
            "used": used,
            "total": total,
            "remaining": remaining,
            "logs": log_map[q.plan_type]
        }

    total_quota = sum(q.total_quota for q in quotas)
    used_quota = sum(site_count_map.get(q.plan_type, 0) for q in quotas)
    remaining_quota = max(total_quota - used_quota, 0)

    # ─────────── 直近28日の「表示回数／クリック数」サイト別ランキング（管理ページと同じ：JST・前日締め）
    # ✅ 統一窓
    start_date, end_date = _gsc_window_by_latest_db(28)
    rank_impr_28d = []
    rank_clicks_28d = []
    if end_date:
        # 表示回数 Top50
        rank_impr_28d = (
            db.session.query(
                Site.id.label("site_id"),
                Site.name.label("site_name"),
                Site.url.label("site_url"),
                User.username.label("username"),
                func.coalesce(func.sum(GSCDailyTotal.impressions), 0).label("value"),
            )
            .join(GSCDailyTotal, GSCDailyTotal.site_id == Site.id)
            .join(User, User.id == Site.user_id)
            .filter(GSCDailyTotal.date >= start_date, GSCDailyTotal.date <= end_date)
            .group_by(Site.id, Site.name, Site.url, User.username)
            .order_by(func.coalesce(func.sum(GSCDailyTotal.impressions), 0).desc())
            .limit(50)
            .all()
        )
        # クリック数 Top50
        rank_clicks_28d = (
            db.session.query(
                Site.id.label("site_id"),
                Site.name.label("site_name"),
                Site.url.label("site_url"),
                User.username.label("username"),
                func.coalesce(func.sum(GSCDailyTotal.clicks), 0).label("value"),
            )
            .join(GSCDailyTotal, GSCDailyTotal.site_id == Site.id)
            .join(User, User.id == Site.user_id)
            .filter(GSCDailyTotal.date >= start_date, GSCDailyTotal.date <= end_date)
            .group_by(Site.id, Site.name, Site.url, User.username)
            .order_by(func.coalesce(func.sum(GSCDailyTotal.clicks), 0).desc())
            .limit(50)
            .all()
        )

    
    return render_template(
        "dashboard.html",
        gsc_win_start=start_date,
        gsc_win_end=end_date,
        plan_type=quotas[0].plan_type if quotas else "未契約",
        total_quota=total_quota,
        used_quota=used_quota,
        remaining_quota=remaining_quota,
        total_articles=g.total_articles,
        done=g.done,
        posted=g.posted,
        error=g.error,
        plans=plans,
        # ▼ ランキング用（テンプレのタブから使用）
        rank_impr_28d=rank_impr_28d,
        rank_clicks_28d=rank_clicks_28d,
    )


# ─────────── Error Details
from app.models import Error  # ← Error モデルを追加
from flask import render_template, request

@bp.route("/<username>/view_errors")
@login_required
def view_errors(username):
    if current_user.username != username:
        abort(403)

    # ページ番号を取得
    page = request.args.get('page', 1, type=int)
    
    # エラー情報の取得（ページネーション対応）
    errors = Error.query.filter_by(user_id=current_user.id).order_by(Error.created_at.desc()).paginate(page=page, per_page=10)

    return render_template(
        "view_errors.html",
        errors=errors  # エラー詳細のリストをテンプレートに渡す
    )



@bp.route("/api/rankings")
@login_required
def api_rankings():
    # 短時間で必ず応答させる安全網（Webだけ。トランザクション内で有効）
    try:
        db.session.execute("SET LOCAL statement_timeout = '5s'")
    except Exception:
        pass

    rank_type = (request.args.get("type") or "site").lower()
    limit = min(max(int(request.args.get("limit", 50)), 1), 50)

    # ---- Redisキャッシュ（キーに“ランキング種別＋期間”を含める）----
    from app import redis_client
    from flask import current_app
    import json
    from datetime import datetime, timezone, timedelta

    JST = timezone(timedelta(hours=9))
    today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
    end_date = today_jst - timedelta(days=1)         # 前日締め
    start_date = end_date - timedelta(days=27)       # 28日窓
    cache_key = f"rankings:{rank_type}:{limit}:{start_date.isoformat()}:{end_date.isoformat()}"

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify(json.loads(cached))
    except Exception as e:
        current_app.logger.warning(f"[rankings] redis GET failed: {e}")

    # ========== 1) ユーザー毎の登録サイト数ランキング ==========
    if rank_type == "site":
        # ここは “サイト数＝Siteの件数” を数えるだけ。重い列は持たない。
        excluded_user_ids = [1, 2, 14, 24]

        # Site.user_id にインデックスが必要（下でコマンド案を出します）
        subq = (
            db.session.query(
                User.id.label("user_id"),
                User.last_name,
                User.first_name,
                func.count(Site.id).label("site_count")
            )
            .filter(~User.id.in_(excluded_user_ids))
            .outerjoin(Site, Site.user_id == User.id)
            .group_by(User.id, User.last_name, User.first_name)
        ).subquery()

        rows = (
            db.session.query(
                subq.c.user_id,
                subq.c.last_name,
                subq.c.first_name,
                subq.c.site_count
            )
            .order_by(subq.c.site_count.desc())
            .limit(limit)
            .all()
        )

        data = [
            {
                "user_id": int(r.user_id),
                "last_name": r.last_name or "",
                "first_name": r.first_name or "",
                "site_count": int(r.site_count or 0),
            }
            for r in rows
        ]

        try:
            redis_client.setex(cache_key, 60, json.dumps(data))  # 60秒キャッシュ
        except Exception as e:
            current_app.logger.warning(f"[rankings] redis SETEX failed: {e}")
        return jsonify(data)

    # ========== 2) 28日合計：サイト別の表示回数 or クリック数 ==========
    # テーブルからは “必要な列だけ” 取り出す。重い列（body, *_prompt など）は参照しない。
    from app.models import GSCDailyTotal

    metric_col = GSCDailyTotal.impressions if rank_type in ("impressions", "impr") else GSCDailyTotal.clicks

    rows = (
        db.session.query(
            Site.id.label("site_id"),
            Site.name.label("site_name"),
            Site.url.label("site_url"),
            User.last_name.label("last_name"),
            User.first_name.label("first_name"),
            User.username.label("username"),
            func.coalesce(func.sum(metric_col), 0).label("value"),
        )
        .join(GSCDailyTotal, GSCDailyTotal.site_id == Site.id)
        .join(User, User.id == Site.user_id)
        .filter(~User.id.in_([1, 14, 24]))  # 追加：ID14・24も除外
        # ランキングから除外するサイト名
        .filter(~Site.name.in_(["天草生うに本舗 丸健水産オンラインショップ"]))
        .filter(~Site.url.like("https://shopping-douko.com%"))
        .filter(GSCDailyTotal.date >= start_date, GSCDailyTotal.date <= end_date)
        .group_by(Site.id, Site.name, Site.url, User.last_name, User.first_name, User.username)
        .order_by(func.coalesce(func.sum(metric_col), 0).desc())
        .limit(limit)
        .all()
    )

    def _display_name(r):
        ln = (r.last_name or "").strip()
        fn = (r.first_name or "").strip()
        full = f"{ln}{fn}"
        return full if full else (r.username or "")

    data = [
        {
            "site_id": r.site_id,
            "site_name": r.site_name,
            "site_url": r.site_url,
            "last_name": r.last_name or "",
            "first_name": r.first_name or "",
            "name": _display_name(r),          # 互換キー
            "display_name": _display_name(r),  # 互換キー
            "username": r.username,
            "value": int(r.value or 0),
        }
        for r in rows
    ]

    try:
        redis_client.setex(cache_key, 60, json.dumps(data))
    except Exception as e:
        current_app.logger.warning(f"[rankings] redis SETEX failed: {e}")
    return jsonify(data)




# ─────────── プロンプト CRUD（新規登録のみ）
@bp.route("/<username>/prompts", methods=["GET", "POST"])
@login_required
def prompts(username):
    if current_user.username != username:
        abort(403)

    form = PromptForm()

    if form.validate_on_submit():
        db.session.add(PromptTemplate(
            genre    = form.genre.data,
            title_pt = form.title_pt.data,
            body_pt  = form.body_pt.data,
            user_id  = current_user.id
        ))
        db.session.commit()
        flash("プロンプトを保存しました", "success")
        return redirect(url_for(".prompts", username=username))

    plist = PromptTemplate.query.filter_by(user_id=current_user.id).all()
    return render_template("prompts.html", form=form, prompts=plist)



# ─────────── プロンプト編集ページ（専用ページ）
@bp.route("/prompt/edit/<int:pid>", methods=["GET", "POST"])
@login_required
def edit_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)

    form = PromptForm(obj=pt)
    if form.validate_on_submit():
        pt.genre    = form.genre.data
        pt.title_pt = form.title_pt.data
        pt.body_pt  = form.body_pt.data
        db.session.commit()
        flash("プロンプトを更新しました", "success")
        return redirect(url_for(".prompts", username=current_user.username))

    return render_template("prompt_edit.html", form=form, prompt=pt)


# ─────────── プロンプト削除
@bp.post("/prompts/delete/<int:pid>")
@login_required
def delete_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    db.session.delete(pt)
    db.session.commit()
    flash("削除しました", "success")
    return redirect(url_for(".prompts", username=current_user.username))


# ─────────── プロンプト取得API（記事生成用）
@bp.route("/api/prompt/<int:pid>")
@login_required
def api_prompt(pid: int):
    pt = PromptTemplate.query.get_or_404(pid)
    if pt.user_id != current_user.id:
        abort(403)
    return jsonify({
        "title_pt": pt.title_pt,
        "body_pt": pt.body_pt
    })

@bp.route("/purchase-history")
@login_required
def purchase_history():
    user = current_user

    # SiteQuotaLogから登録枠履歴（すべてのプラン分）
    logs = SiteQuotaLog.query.filter_by(user_id=user.id).order_by(SiteQuotaLog.created_at.desc()).all()

    return render_template("purchase_history.html", logs=logs)


# ──────────────── NEW: ユーザー用 インデックス申請（UI補助） ────────────────
@bp.route("/<username>/index-monitor")
@login_required
def index_monitor(username):
    """
    自分のサイトのインデックス状況サマリ（直近28日）＋
    申請UI補助（GSCで開くボタン）を最速表示。
    ※GSC APIは叩かず、DBの集計値のみを使う（1秒以内）。
    """
    from datetime import date, timedelta
    from app.models import Site, Article, GSCDailyTotal, GSCConfig, User

    # 認可：自分のページのみ（管理者の代理ログインは既存ロジックに準拠）
    if current_user.username != username and not (getattr(current_user, "is_admin", False) or session.get("admin_id")):
        flash("権限がありません。", "danger")
        return redirect(url_for("main.dashboard", username=current_user.username))

    date_28d_ago = date.today() - timedelta(days=28)

    # 対象サイト（自ユーザーのサイトのみ）
    sites = Site.query.filter_by(user_id=current_user.id).order_by(Site.id.asc()).all()
    site_ids = [s.id for s in sites]

    # 最新の GSCConfig（property_uri）をサイトごとに1件取得するサブクエリ
    # （高速：サイト数が多くても1クエリで取る）
    sub_cfg_max = (
        db.session.query(
            GSCConfig.site_id,
            func.max(GSCConfig.id).label("max_id")
        )
        .filter(GSCConfig.site_id.in_(site_ids))
        .group_by(GSCConfig.site_id)
        .subquery()
    )
    latest_cfg = {
        cfg.site_id: cfg.property_uri
        for cfg in db.session.query(GSCConfig)
                .join(sub_cfg_max, (GSCConfig.site_id == sub_cfg_max.c.site_id) & (GSCConfig.id == sub_cfg_max.c.max_id))
                .all()
    } if site_ids else {}

    # 直近28日間のGSC掲載（サイト単位で何日分の行があるか）= 掲載の“強さ”近似
    sub_gsc = (
        db.session.query(
            GSCDailyTotal.site_id,
            func.count(GSCDailyTotal.id).label("indexed_days")  # 表示のあった日数近似
        )
        .filter(GSCDailyTotal.site_id.in_(site_ids), GSCDailyTotal.date >= date_28d_ago)
        .group_by(GSCDailyTotal.site_id)
        .subquery()
    )

    # サイト別サマリ（記事数・掲載日数近似・率）
    summary = []
    for s in sites:
        total_articles = db.session.query(func.count(Article.id)).filter(Article.site_id == s.id).scalar() or 0
        # 左側の FROM を sub_gsc に明示し、該当 site_id で絞り込む（JOIN の曖昧さを解消）
        indexed_days = (
            db.session.query(func.coalesce(sub_gsc.c.indexed_days, 0))
            .select_from(sub_gsc)
            .filter(sub_gsc.c.site_id == s.id)
            .scalar() or 0
        )
        rate = round((indexed_days / 28.0) * 100.0, 1) if 28 > 0 else 0.0
        summary.append({
            "site_id": s.id,
            "site_url": s.url,
            "gsc_connected": s.gsc_connected,
            "article_count": total_articles,
            "indexed_days": int(indexed_days),
            "rate": rate,
            "property_uri": latest_cfg.get(s.id)  # GSC検査URLを作るのに使用
        })

    # gsc_url_status に「indexed=TRUE」の行がある URL は除外し、
    # FALSE または NULL（＝未インデックス/未検査相当）のみを出す。
    # 結合キーは article_id が入っていればそれを優先し、なければ (site_id, url) でマッチ。
    from app.models import GSCUrlStatus  # 既にインポート済みならこの行は自動的に冗長だが無害

    recent_articles = (
        db.session.query(
            Article.id, Article.title, Article.posted_url, Article.site_id, Article.posted_at
        )
        # URLステータスと外部結合（重複レコードがあり得るため、後段でGROUP BY）
        .outerjoin(
            GSCUrlStatus,
            (
                (GSCUrlStatus.article_id == Article.id)
                | (
                    (GSCUrlStatus.site_id == Article.site_id)
                    & (GSCUrlStatus.url == Article.posted_url)
                )
            ),
        )
        .filter(
            Article.site_id.in_(site_ids),
            Article.posted_url.isnot(None),
            ((GSCUrlStatus.indexed == False) | (GSCUrlStatus.indexed.is_(None))),
        )
        # 重複を防ぐ（PostgreSQL互換）：表示列でグルーピング
        .group_by(Article.id, Article.title, Article.posted_url, Article.site_id, Article.posted_at)
        .order_by(Article.posted_at.desc().nullslast(), Article.id.desc())
        .limit(50)
        .all()
    )

    # 🔧 検査URL生成ロジック（案B: サーバ側で正確に構築）
    from urllib.parse import quote

    for art in recent_articles:
        prop = latest_cfg.get(art.site_id)
        if not prop:
            inspect_url = None
        else:
            # property_uri が sc-domain: か URL プレフィックスかで分岐
            # ドメインプロパティは非エンコード、URLプレフィックスは : / を保持して渡す
            if prop.startswith("sc-domain:"):
                resource_id = prop  # e.g. sc-domain:example.com
            else:
                p = prop if prop.endswith("/") else (prop + "/")
                resource_id = quote(p, safe=":/")  # 実質そのまま、: と / は保持

            url_encoded = quote(art.posted_url or "", safe="")
            inspect_url = (
                f"https://search.google.com/search-console/inspect"
                f"?resource_id={resource_id}&url={url_encoded}&page=inspect"
            )
        # Row は不変なので、6要素タプルへ詰め替える
        # (id, title, url, site_id, posted_at, inspect_url)
        # 後段テンプレでこの順序をそのまま使う
        pass

    # ↑の pass は for ループを抜けるためのプレースホルダではないので注意。
    # 実際には recent_articles を新しい配列に詰め替える：
    from urllib.parse import quote  # 念のためスコープ維持
    recent_articles_with_inspect = []
    for (aid, title, url, site_id, posted_at) in recent_articles:
        prop = latest_cfg.get(site_id)
        if prop and url:
            if prop.startswith("sc-domain:"):
                resource_id = prop  # 非エンコードでそのまま
            else:
                p = prop if prop.endswith("/") else (prop + "/")
                resource_id = quote(p, safe=":/")  # : / を保持
            inspect_url = (
                "https://search.google.com/search-console/inspect"
                "?resource_id={}&url={}&page=inspect"
            ).format(resource_id, quote(url, safe=""))
        else:
            inspect_url = None
        recent_articles_with_inspect.append((aid, title, url, site_id, posted_at, inspect_url))
 

    return render_template(
        "index_monitor.html",
        summary=summary,
        recent_articles=recent_articles_with_inspect,
        username=username,
    )


# ────────────── 登録サイト管理 ──────────────

from os import getenv
from app.forms import SiteForm
from app.models import SiteQuotaLog
from app.services.internal_seo.enqueue import enqueue_new_site
from app.services.internal_seo.applier import preview_apply_for_post
from flask import render_template

@bp.route("/<username>/sites", methods=["GET", "POST"])
@login_required
def sites(username):
    if current_user.username != username:
        abort(403)

    form = SiteForm()
    user = current_user

    # ✅ ジャンルの選択肢をセット（自分が追加したジャンルのみ）
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    form.genre_id.choices = [(0, "ジャンル未選択")] + [(g.id, g.name) for g in genre_list]

    # 🔹 登録済みサイト一覧と件数
    site_list = Site.query.filter_by(user_id=user.id).all()

    # 🔹 プランごとのクォータデータ
    quotas = UserSiteQuota.query.filter_by(user_id=user.id).all()

    # 🔹 プランごとのリアルタイム使用状況と履歴ログ
    quota_by_plan = {}
    for q in quotas:
        plan = q.plan_type
        used = Site.query.filter_by(user_id=user.id, plan_type=plan).count()  # ← 🔄 used_quotaをリアルタイムで算出
        total = q.total_quota or 0
        remaining = max(total - used, 0)
        logs = SiteQuotaLog.query.filter_by(user_id=user.id, plan_type=plan).order_by(SiteQuotaLog.created_at.desc()).all()

        quota_by_plan[plan] = {
            "total": total,
            "used": used,
            "remaining": remaining,
            "logs": logs
        }

    # 🔹 全体のトータル数と残数もリアルタイムで統一
    total_quota = sum([q.total_quota for q in quotas])
    used_quota = sum([Site.query.filter_by(user_id=user.id, plan_type=q.plan_type).count() for q in quotas])
    remaining_quota = total_quota - used_quota

    if form.validate_on_submit():
        if used_quota >= total_quota:
            flash("サイト登録上限に達しています。追加するには課金が必要です。", "danger")
            return redirect(url_for("main.sites", username=username))

        selected_plan = form.plan_type.data
        quota = UserSiteQuota.query.filter_by(user_id=user.id, plan_type=selected_plan).first()
        if quota:
            quota.used_quota = Site.query.filter_by(user_id=user.id, plan_type=selected_plan).count() + 1  # 🔄更新（念のため）
        else:
            flash("プラン情報が見つかりません。", "danger")
            return redirect(url_for("main.sites", username=username))

        # ① まず作成してIDを確定
        new_site = Site(
            name       = form.name.data,
            url        = form.url.data.rstrip("/"),
            username   = form.username.data,
            app_pass   = form.app_pass.data,
            user_id    = user.id,
            plan_type  = selected_plan,
            genre_id   = form.genre_id.data if form.genre_id.data != 0 else None,  # ✅
        )
        db.session.add(new_site)
        db.session.commit()  # ← ここで new_site.id が確定

        # ② 登録直後に内部SEOをenqueue（非同期ワーカーが拾う）
        try:
            enqueue_new_site(new_site.id)
            flash("サイトを登録しました（内部SEOの初期処理を開始しました）", "success")
        except Exception as e:
            # enqueue に失敗してもサイト登録自体は成功として扱う（既存機能を壊さない）
            current_app.logger.exception(f"[internal-seo] enqueue failed on site create: {e}")
            flash("サイトを登録しました（内部SEO初期処理の登録に失敗しました。後からやり直せます）", "warning")
        return redirect(url_for("main.sites", username=username))

    # 🔹 Stripe履歴（参考表示用）
    history_logs = PaymentLog.query.filter_by(user_id=user.id).order_by(PaymentLog.created_at.desc()).all()
# 🔍 最初に優先表示するプラン（business優先）
    # 例：affiliate を優先して初期表示にする
    default_plan = "affiliate" if "affiliate" in quota_by_plan else "business"


    return render_template(
        "sites.html",
        form=form,
        sites=site_list,
        plans=quota_by_plan,
        remaining_quota=remaining_quota,  # ✅ ← 左上の表示に使用
        total_quota=total_quota,
        used_quota=used_quota,
        history_logs=history_logs,
        stripe_public_key=os.getenv("STRIPE_PUBLIC_KEY"),
        default_plan=default_plan  # ← 🔥追加！
    )


@bp.post("/<username>/sites/<int:sid>/delete")
@login_required
def delete_site(username, sid: int):
    if current_user.username != username:
        abort(403)

    site = Site.query.get_or_404(sid)
    if site.user_id != current_user.id:
        abort(403)

    db.session.delete(site)
    db.session.commit()
    flash("サイトを削除しました", "success")
    return redirect(url_for("main.sites", username=username))


@bp.route("/<username>/sites/<int:sid>/edit", methods=["GET", "POST"])
@login_required
def edit_site(username, sid: int):
    if current_user.username != username:
        abort(403)

    site = Site.query.get_or_404(sid)
    if site.user_id != current_user.id:
        abort(403)

    form = SiteForm(obj=site)

    # ✅ 自分のジャンルだけを選択肢に含める
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    form.genre_id.choices = [(0, "ジャンル未選択")] + [(g.id, g.name) for g in genre_list]

    # ✅ 初期値は GET のときだけ設定（POST時に上書きしない！）
    if request.method == "GET":
        form.genre_id.data = site.genre_id if site.genre_id is not None else 0

    if form.validate_on_submit():
        site.name       = form.name.data
        site.url        = form.url.data.rstrip("/")
        site.username   = form.username.data
        site.app_pass   = form.app_pass.data
        site.plan_type  = form.plan_type.data
        site.genre_id   = form.genre_id.data if form.genre_id.data != 0 else None

        db.session.commit()
        flash("サイト情報を更新しました", "success")
        return redirect(url_for("main.log_sites", username=username))

    else:
        if request.method == "POST":
            print("❌ バリデーションエラー:", form.errors)
            print("📌 ジャンルID:", form.genre_id.data)

    return render_template("site_edit.html", form=form, site=site)

@bp.route('/add_genre', methods=['POST'])
@login_required
def add_genre():
    data = request.get_json()
    name = data.get('name')
    description = data.get('description', '')

    if not name:
        return jsonify(success=False, error='Name required'), 400

    new_genre = Genre(name=name, description=description, user_id=current_user.id)
    db.session.add(new_genre)
    db.session.commit()

    return jsonify(success=True, genre_id=new_genre.id, genre_name=new_genre.name)


# ─────────── 記事生成（ユーザー別）

@bp.route("/<username>/generate", methods=["GET", "POST"])
@login_required
def generate(username):
    if current_user.username != username:
        abort(403)

    form = GenerateForm()

    # ▼ プロンプトとサイトの選択肢をセット
    form.genre_select.choices = [(0, "― 使わない ―")] + [
        (p.id, p.genre)
        for p in PromptTemplate.query.filter_by(user_id=current_user.id)
    ]
    form.site_select.choices = [(0, "―― 選択 ――")] + [
        (s.id, s.name)
        for s in Site.query.filter_by(user_id=current_user.id)
    ]

    # ▼ クエリパラメータから事前選択されたsite_idとstatusを取得
    selected_site_id = request.args.get("site_id", type=int)
    status_filter = request.args.get("status")  # "used" / "unused" / None

    if request.method == "GET" and selected_site_id:
        form.site_select.data = selected_site_id

    # ▼ POST処理（記事生成）
    if form.validate_on_submit():
        kws = [k.strip() for k in form.keywords.data.splitlines() if k.strip()]
        site_id = form.site_select.data or None
        enqueue_generation(
            current_user.id,
            kws,
            form.title_prompt.data,
            form.body_prompt.data,
            site_id
        )
        flash(f"{len(kws)} 件をキューに登録しました", "success")
        return redirect(url_for("main.log_sites", username=username))

    # ▼ 表示するキーワード一覧を取得（statusフィルタも考慮）
    keyword_choices = []
    selected_site = None
    site_name = None

    # ▼ 件数カウント用の変数を初期化（デフォルトは0）
    total_count = 0
    used_count = 0
    unused_count = 0

    if form.site_select.data:
        selected_site_id = form.site_select.data
        selected_site = Site.query.get(selected_site_id)
        site_name = selected_site.name if selected_site else ""

        keyword_query = Keyword.query.filter_by(
            user_id=current_user.id,
            site_id=selected_site_id
        )

        # ▼ 件数カウント（フィルタなしで取得）
        all_keywords = keyword_query.all()
        total_count = len(all_keywords)
        used_count = sum(1 for kw in all_keywords if kw.used)
        unused_count = total_count - used_count

        if status_filter == "used":
            keyword_query = keyword_query.filter_by(used=True)
        elif status_filter == "unused":
            keyword_query = keyword_query.filter_by(used=False)

        keyword_choices = keyword_query.order_by(Keyword.id.desc()).limit(1000).all()

    return render_template(
        "generate.html",
        form=form,
        keyword_choices=keyword_choices,
        selected_site=selected_site,
        site_name=site_name,
        status_filter=status_filter,
        total_count=total_count,      # ← 全体件数
        used_count=used_count,        # ← 生成済み件数
        unused_count=unused_count     # ← 未生成件数
    )

# ─────────── GSCルートコード

from app.google_client import fetch_search_queries_for_site
from app.models import Keyword  # 🔁 既存キーワード参照のため追加
from app.article_generator import enqueue_generation  # 🔁 忘れずに

#@bp.route("/generate_from_gsc/<int:site_id>", methods=["GET", "POST"])
#@login_required
#def generate_from_gsc(site_id):
    #site = Site.query.get_or_404(site_id)
    #if site.user_id != current_user.id:
       # abort(403)

    # ✅ GSC未接続のガード
    #if not site.gsc_connected:
        #flash("このサイトはまだSearch Consoleと接続されていません。", "danger")
        #return redirect(url_for("main.gsc_connect"))

    #try:
        #rows = fetch_search_queries(site.url, days=7, row_limit=40)
        #keywords = [row["keys"][0] for row in rows if "keys" in row]
    #except Exception as e:
        #flash(f"Search Consoleからキーワードの取得に失敗しました: {e}", "danger")
        #return redirect(url_for("main.keywords", username=current_user.username))

    #if not keywords:
        #flash("検索クエリが見つかりませんでした。", "warning")
        #return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ 既存キーワードの重複チェック
    #existing_keywords = set(
        #k.keyword for k in Keyword.query.filter_by(site_id=site.id).all()
    #)
    #new_keywords = [kw for kw in keywords if kw not in existing_keywords]

    #if not new_keywords:
        #flash("すべてのキーワードが既に登録されています。", "info")
        #return redirect(url_for("main.keywords", username=current_user.username))

    # ✅ GSC由来のキーワードとしてDBに追加
    #for kw in new_keywords:
        #db.session.add(Keyword(
            #keyword=kw,
            #site_id=site.id,
            #user_id=current_user.id,
            #source='gsc'
        #))

    # ✅ GSC接続状態を保存（初回のみ）※保険として残す
    #if not site.gsc_connected:
        #site.gsc_connected = True

    #db.session.commit()

    # ✅ 記事生成キューへ
    #enqueue_generation(new_keywords, site.id, current_user.id)

    #flash(f"{len(new_keywords)}件のキーワードから記事生成を開始しました", "success")
    #return redirect(url_for("main.keywords", username=current_user.username))


@bp.route("/gsc_generate", methods=["GET", "POST"])
@login_required
def gsc_generate():
    from app.google_client import fetch_search_queries_for_site
    from app.article_generator import enqueue_generation
    from app.models import Keyword, PromptTemplate

    # --- POST（記事生成処理） ---
    if request.method == "POST":
        site_id = request.form.get("site_id", type=int)
        site = Site.query.get_or_404(site_id)

        if site.user_id != current_user.id:
            abort(403)

        # ✅ 追加：すでにGSC生成が始まっている場合は中止
        if site.gsc_generation_started:
            flash("⚠️ このサイトではすでにGSC記事生成が開始されています。", "warning")
            return redirect(url_for("main.gsc_generate", site_id=site_id))

        # ✅ 初回生成フラグをTrueにする（1回限りの起動）
        site.gsc_generation_started = True
        db.session.commit()

        prompt_id = request.form.get("prompt_id", type=int)
        title_prompt = request.form.get("title_prompt", "").strip()
        body_prompt = request.form.get("body_prompt", "").strip()

        if not site.gsc_connected:
            flash("このサイトはまだGSCと接続されていません。", "danger")
            return redirect(url_for("main.gsc_connect"))


        # GSCクエリ取得
        try:
            queries = fetch_search_queries_for_site(site, days=28, row_limit=1000)

            # 🔧 追加: 取得件数ログ
            current_app.logger.info(f"[GSC] {len(queries)} 件のクエリを取得 - {site.url}")

        except Exception as e:
            flash(f"GSCからのクエリ取得に失敗しました: {e}", "danger")
            return redirect(url_for("main.log_sites", username=current_user.username))

        # 重複排除
        # ✅ 既存キーワードのうち、status="done" のものは再利用不可として除外
        existing = set(
            k.keyword
            for k in Keyword.query.filter_by(site_id=site.id, source="gsc")
            if k.status == "done"
        )
        new_keywords = [q for q in queries if q not in existing]

        # 🔧 追加: 空 or 全重複の分岐で別メッセージ
        if not new_keywords:
            if not queries:
                flash("⚠️ GSCからクエリを取得できませんでした。URL形式が一致していない可能性があります。", "warning")
                current_app.logger.warning(f"[GSC] クエリが0件でした - {site.url}")
            else:
                flash("すべてのクエリが既に登録されています。", "info")
                current_app.logger.info(f"[GSC] 全クエリが既存のため登録スキップ - {site.url}")
            return redirect(url_for("main.log_sites", username=current_user.username))

        # DBに登録（source='gsc'）
        for kw in new_keywords:
            keyword = Keyword(site_id=site.id, keyword=kw, user_id=current_user.id, source="gsc")
            db.session.add(keyword)
        db.session.commit()

        # 🔸プロンプト取得（保存済みを優先）
        if prompt_id:
            saved_prompt = PromptTemplate.query.filter_by(id=prompt_id, user_id=current_user.id).first()
            if saved_prompt:
                title_prompt = saved_prompt.title_pt
                body_prompt = saved_prompt.body_pt

        # 🔁 記事生成をキューに追加
        enqueue_generation(
            user_id=current_user.id,
            site_id=site.id,
            keywords=new_keywords,
            title_prompt=title_prompt,
            body_prompt=body_prompt,
        )

        flash(f"{len(new_keywords)}件のGSCキーワードから記事生成を開始しました", "success")
        current_app.logger.info(f"[GSC] ✅ {len(new_keywords)} 件の記事生成キューを追加 - {site.url}")
        return redirect(url_for("main.log_sites", username=current_user.username))

    # --- GET（フォーム表示） ---
    site_id = request.args.get("site_id", type=int)
    if not site_id:
        flash("サイトIDが指定されていません。", "danger")
        return redirect(url_for("main.log_sites", username=current_user.username))

    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        abort(403)

    if not site.gsc_connected:
        flash("このサイトはまだGSCと接続されていません。", "danger")
        return redirect(url_for("main.gsc_connect"))
    
    # ✅ 追加: ステータスでフィルタリング
    status_filter = request.args.get("status")
    query = Keyword.query.filter_by(site_id=site.id, source="gsc")

    if status_filter in ["done", "unprocessed"]:
        query = query.filter(Keyword.status == status_filter)

    from app.models import Article, Keyword

# ✅ GSC由来の記事数（Keyword.source="gsc" に紐づく Article）
    # ✅ GSC記事数（JOIN ON 条件を明示）
    gsc_done = Article.query.filter_by(site_id=site.id, source="gsc").count()

# ✅ 全記事数（すべての Article）
    all_done = Article.query.filter_by(site_id=site.id).count()

# ✅ 通常記事数 = 全体 - GSC
    manual_done = all_done - gsc_done

# ✅ 合計・残り（上限：1000）
    total_done = gsc_done + manual_done
    remaining = max(1000 - total_done, 0)
    
    
    # ✅ フィルター前に全GSCキーワードを取得（件数用）
    gsc_done_keywords = Keyword.query.filter_by(site_id=site.id, source="gsc", status="done").count()
    gsc_pending_keywords = Keyword.query.filter_by(site_id=site.id, source="gsc", status="unprocessed").count()
    gsc_total_keywords = gsc_done_keywords + gsc_pending_keywords  # 🔧 合計を追加

    # ✅ 表示リスト用に再フィルタリング
    query = Keyword.query.filter_by(site_id=site.id, source="gsc")
    if status_filter == "done":
        query = query.filter(Keyword.status == "done")
    elif status_filter == "unprocessed":
        query = query.filter(Keyword.status != "done")
    gsc_keywords = query.order_by(Keyword.created_at.desc()).all()


    # 保存済みプロンプト
    saved_prompts = PromptTemplate.query.filter_by(user_id=current_user.id).order_by(PromptTemplate.genre).all()

    return render_template(
        "gsc_generate.html",
        selected_site=site,
        gsc_keywords=gsc_keywords,
        saved_prompts=saved_prompts,
        title_prompt="",  # 初期値
        body_prompt="",   # 初期値
        request=request,   # ✅ テンプレートでセレクトボックス選択保持に使う
        gsc_done=gsc_done,
        manual_done=manual_done,
        total_done=total_done,
        remaining=remaining,
        gsc_done_keywords=gsc_done_keywords,         # ✅ 追加
        gsc_pending_keywords=gsc_pending_keywords,    # ✅ 追加
        gsc_total_keywords=gsc_total_keywords  # 🔧 追加
    )


# --- 既存インポートの下に追加（必要に応じて） ---
from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import Site, db

# ✅ /gsc-connect: GSC連携ページの表示
# ✅ /gsc-connect: GSC連携ページの表示（トークンの有無で判定）
@bp.route("/gsc-connect")
@login_required
def gsc_connect():
    filter_status = request.args.get("status")  # "connected", "unconnected", "all"
    search_query = request.args.get("query", "").strip().lower()
    order = request.args.get("order")  # "recent", "most_views", "least_views"

    # ✅ クエリ構築（全件ベースで始める）
    sites_query = Site.query.filter_by(user_id=current_user.id)

    # ✅ 並び替え条件
    # ※ GSCは「JSTの昨日まで」の直近28日で並べる（相関サブクエリで高速）
    if order in ("most_views", "least_views"):
        from datetime import datetime, timezone, timedelta
        from sqlalchemy import func
        from app.models import GSCDailyTotal
        # ✅ 統一窓
        _start_d, _end_d = _gsc_window_by_latest_db(28)
        _gsc_impr_28d = (
            db.session.query(func.coalesce(func.sum(GSCDailyTotal.impressions), 0))
            .filter(
                GSCDailyTotal.site_id == Site.id,
                GSCDailyTotal.date >= _start_d,
                GSCDailyTotal.date <= _end_d
            )
            .correlate(Site).scalar_subquery()
        )
        if order == "most_views":
            sites_query = sites_query.order_by(_gsc_impr_28d.desc())
        else:  # "least_views"
            sites_query = sites_query.order_by(_gsc_impr_28d.asc())
    else:
        sites_query = sites_query.order_by(Site.created_at.desc())  # デフォルト：新しい順

    sites = sites_query.all()

    # トークン取得
    from app.models import GSCAuthToken
    tokens = {token.site_id: token for token in GSCAuthToken.query.filter_by(user_id=current_user.id).all()}

    # ステータスフラグ付与
    for site in sites:
        site.is_token_connected = site.id in tokens

    # ✅ ステータスフィルター（Python側で処理）
    if filter_status == "connected":
        sites = [s for s in sites if s.gsc_connected]
    elif filter_status == "unconnected":
        sites = [s for s in sites if not s.gsc_connected]

    # ✅ 検索フィルター
    if search_query:
        sites = [s for s in sites if search_query in s.name.lower() or search_query in s.url.lower()]

    return render_template(
        "gsc_connect.html",
        sites=sites,
        filter_status=filter_status,
        search_query=search_query,
        order=order,
    )



@bp.route("/connect_gsc/<int:site_id>", methods=["POST"])
@login_required
def connect_gsc(site_id):
    site = Site.query.get_or_404(site_id)
    if site.user_id != current_user.id:
        flash("アクセス権がありません。", "danger")
        return redirect(url_for("main.gsc_connect"))

    site.gsc_connected = True
    db.session.commit()

    flash(f"✅ サイト「{site.name}」とGoogleサーチコンソールの接続が完了しました。", "success")
    return redirect(url_for("main.gsc_connect"))

# app/routes.py（末尾に追加）

# ✅ 必要なインポート
from flask import request, render_template  # ← Flaskの標準関数
from app.models import GSCMetric, Site      # ← GSCMetricを使ってDBから集計
from flask_login import login_required, current_user
from datetime import datetime, timedelta

# ✅ GSCアクセス分析ルート（ユーザー名不要に統一）
@bp.route("/gsc/<int:site_id>")  # ← ✅ ここを使用ルートに統一
@login_required
def gsc_analysis(site_id):
    # ✅ 対象ユーザーのサイトか確認
    site = Site.query.filter_by(id=site_id, user_id=current_user.id).first_or_404()

    # ✅ 未連携サイトは警告表示
    if not site.gsc_connected:
        return render_template("gsc_analysis.html", site=site, error="このサイトはGSCと未連携です")

    # ✅ GETパラメータ取得（range または start/end）
    range_param = request.args.get("range", "28d")
    start_param = request.args.get("start")
    end_param = request.args.get("end")

    today = datetime.utcnow().date()

    # ✅ 日付範囲の決定ロジック
    if range_param == "1d":
        start_date = today - timedelta(days=1)
    elif range_param == "7d":
        start_date = today - timedelta(days=7)
    elif range_param == "28d":
        start_date = today - timedelta(days=28)
    elif range_param == "3m":
        start_date = today - timedelta(days=90)
    elif range_param == "6m":
        start_date = today - timedelta(days=180)
    elif range_param == "12m":
        start_date = today - timedelta(days=365)
    elif range_param == "16m":
        start_date = today - timedelta(days=480)
    elif range_param == "custom" and start_param and end_param:
        try:
            start_date = datetime.strptime(start_param, "%Y-%m-%d").date()
            today = datetime.strptime(end_param, "%Y-%m-%d").date()
        except ValueError:
            return render_template(
                "gsc_analysis.html",
                site=site,
                error="日付形式が不正です"
            )
    else:
        # ✅ デフォルト28日
        start_date = today - timedelta(days=28)

    # ✅ データベースから該当期間のGSCメトリクスを取得
    metrics = GSCMetric.query.filter(
        GSCMetric.site_id == site_id,
        GSCMetric.date >= start_date,
        GSCMetric.date <= today
    ).order_by(GSCMetric.date.asc()).all()

    # ✅ テンプレートへデータ送信
    return render_template(
        "gsc_analysis.html",
        site=site,
        metrics=metrics,
        start_date=start_date,
        end_date=today,
        selected_range=range_param
    )


# ─────────── 生成ログ
@bp.route("/<username>/log/site/<int:site_id>")
@login_required
def log(username, site_id):
    if current_user.username != username:
        abort(403)

    from collections import defaultdict
    from .article_generator import _generate_slots_per_site

    # ステータス & ソートキー取得
    status = request.args.get("status")
    sort_key = request.args.get("sort", "scheduled_at")
    sort_order = request.args.get("order", "desc")

    # ✅ GSC絞り込み用パラメータ取得
    source = request.args.get("source", "all")

    # 未スケジュール記事の slot を自動割当
    unscheduled = Article.query.filter(
        Article.user_id == current_user.id,
        Article.scheduled_at.is_(None),
    ).all()

    if unscheduled:
        site_map = defaultdict(list)
        for art in unscheduled:
            if art.site_id:
                site_map[art.site_id].append(art)

        for sid, articles in site_map.items():
            slots = iter(_generate_slots_per_site(current_app, sid, len(articles)))
            for art in articles:
                art.scheduled_at = next(slots)
        db.session.commit()

    # 記事取得クエリ
    q = Article.query.filter_by(user_id=current_user.id, site_id=site_id)
    if status:
        q = q.filter_by(status=status)

    if source == "gsc":
        q = q.filter_by(source="gsc")  # ✅ GSC記事のみフィルタ

    # 必ず site 情報も preload（clicks/impressions用）
    q = q.options(selectinload(Article.site))

    # 初期並び順：投稿予定日時優先
    q = q.order_by(
        nulls_last(asc(Article.scheduled_at)),
        Article.created_at.desc(),
    )

    articles = q.all()

    # --- 当該サイトの直近28日合計（JST）を取得（記事行の表示＆並べ替え用） ---
    # ✅ 統一窓（JSTの昨日 ∧ DB最新日）
    start_d, end_d = _gsc_window_by_latest_db(28)

    gsc_row = (
        db.session.query(
            func.coalesce(func.sum(GSCDailyTotal.clicks), 0),
            func.coalesce(func.sum(GSCDailyTotal.impressions), 0),
        )
        .filter(
            GSCDailyTotal.site_id == site_id,
            GSCDailyTotal.date >= start_d,
            GSCDailyTotal.date <= end_d,
        )
        .first()
    )
    site_gsc = {
        "clicks": int(gsc_row[0] or 0),
        "impressions": int(gsc_row[1] or 0),
    }

    # 🔽 並び替え（Python側）: クリック/表示回数はサイト合計でソート
    if sort_key == "clicks":
        keyval = site_gsc["clicks"]
        articles.sort(key=lambda _a: keyval, reverse=(sort_order == "desc"))
    elif sort_key == "impr":
        keyval = site_gsc["impressions"]
        articles.sort(key=lambda _a: keyval, reverse=(sort_order == "desc"))

    site = Site.query.get_or_404(site_id)

    return render_template(
        "log.html",
        articles=articles,
        site=site,
        status=status,
        sort_key=sort_key,
        sort_order=sort_order,
        selected_source=source,  # ✅ フィルタUIの状態保持用
        jst=JST,
        site_gsc=site_gsc,  # ✅ 追加: テンプレートへ渡す
    )



# ─────────── ログ：サイト選択ページ（ユーザー別）
@bp.route("/<username>/log/sites")
@login_required
def log_sites(username):
    if current_user.username != username:
        abort(403)

    from sqlalchemy import case
    from app.models import Genre, GSCDailyTotal
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func, asc, desc
    from sqlalchemy.orm import selectinload


    # GETパラメータ
    status_filter = request.args.get("plan_type", "all")
    search_query = request.args.get("query", "").strip().lower()
    sort_key = request.args.get("sort", "created")
    sort_order = request.args.get("order", "asc")
    genre_id = request.args.get("genre_id", "0")
    try:
        genre_id = int(genre_id)
    except ValueError:
        genre_id = 0

    # ---------- GSC合計（直近28日・JST） ----------
    JST = timezone(timedelta(hours=9))
    today_jst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(JST).date()
    # ✅ GSC UI と同じ「昨日までの28日」
    end_d   = today_jst - timedelta(days=1)
    start_d = end_d - timedelta(days=27)

    gsc_sub = (
        db.session.query(
            GSCDailyTotal.site_id.label("site_id"),
            func.coalesce(func.sum(GSCDailyTotal.clicks), 0).label("clicks"),
            func.coalesce(func.sum(GSCDailyTotal.impressions), 0).label("impressions"),
        )
        .filter(GSCDailyTotal.date >= start_d, GSCDailyTotal.date <= end_d)
        .group_by(GSCDailyTotal.site_id)
    ).subquery()

    # ---------- サブクエリ（記事数などの集計＋GSC合計をJOIN） ----------
    subquery = (
        db.session.query(
            Site.id.label("id"),
            Site.name.label("name"),
            Site.url.label("url"),
            Site.plan_type.label("plan_type"),
            Site.gsc_connected.label("gsc_connected"),
            Site.created_at.label("created_at"),
            func.count(Article.id).label("total"),
            func.sum(case((Article.status == "done", 1), else_=0)).label("done"),
            func.sum(case((Article.status == "posted", 1), else_=0)).label("posted"),
            func.sum(case((Article.status == "error", 1), else_=0)).label("error"),
            func.coalesce(func.max(gsc_sub.c.clicks), 0).label("clicks"),
            func.coalesce(func.max(gsc_sub.c.impressions), 0).label("impressions"),
        )
        .select_from(Site)  # ← 左側（FROM）を明示して暗黙JOINの曖昧さを解消
        .outerjoin(Article, Site.id == Article.site_id)
        .outerjoin(gsc_sub, gsc_sub.c.site_id == Site.id)
        .filter(Site.user_id == current_user.id)
        .group_by(Site.id)
    ).subquery()

    # ---------- メインクエリ（フィルター＆並び替え） ----------
    query = db.session.query(subquery)

    if status_filter in ["affiliate", "business"]:
        query = query.filter(subquery.c.plan_type == status_filter)

    if genre_id > 0:
        # サブクエリを左側に固定してから Site をJOIN（ON句は既に明示）
        query = (
            query.select_from(subquery).join(Site, Site.id == subquery.c.id).filter(Site.genre_id == genre_id)
        )

    if search_query:
        query = query.filter(
            func.lower(subquery.c.name).like(f"%{search_query}%") |
            func.lower(subquery.c.url).like(f"%{search_query}%")
        )

    # 並び順カラム設定
    sort_columns = {
        "created": subquery.c.created_at,
        "total": subquery.c.total,
        "done": subquery.c.done,
        "posted": subquery.c.posted,
        "clicks": subquery.c.clicks,
        "impressions": subquery.c.impressions
    }
    order_column = sort_columns.get(sort_key, subquery.c.created_at)

    if sort_order == "desc":
        query = query.order_by(order_column.desc())
    else:
        query = query.order_by(order_column.asc())

    result = query.all()

    # ジャンル一覧
    genre_list = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    genre_choices = [(0, "すべてのジャンル")] + [(g.id, g.name) for g in genre_list]

    return render_template(
        "log_sites.html",
        sites=result,
        selected_status=status_filter,
        selected_genre_id=genre_id,
        genre_choices=genre_choices,
        search_query=search_query,
        sort_key=sort_key,
        sort_order=sort_order
    )


# ─────────── プレビュー
@bp.route("/preview/<int:article_id>")
@login_required
def preview(article_id: int):
    art = Article.query.get_or_404(article_id)
    if art.user_id != current_user.id:
        abort(403)
    styled = _decorate_html(art.body or "")
    return render_template("preview.html", article=art, styled_body=styled)


# ─────────── WordPress 即時投稿
@bp.post("/article/<int:id>/post")
@login_required
def post_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    if not art.site:
        flash("投稿先サイトが設定されていません", "danger")
        return redirect(url_for(".log", site_id=art.site_id))

    try:
        url = post_to_wp(art.site, art)
        art.posted_at = datetime.utcnow()
        art.status = "posted"
        db.session.commit()
        flash(f"WordPress へ投稿しました: {url}", "success")
    except Exception as e:
        current_app.logger.exception("即時投稿失敗: %s", e)
        db.session.rollback()
        flash(f"投稿失敗: {e}", "danger")

    return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))


# ─────────── 記事編集・削除・再試行
@bp.route("/article/<int:id>/edit", methods=["GET", "POST"])
@login_required
def edit_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    form = ArticleForm(obj=art)
    if form.validate_on_submit():
        art.title = form.title.data
        art.body  = form.body.data
        db.session.commit()
        flash("記事を更新しました", "success")
        return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))
    return render_template("edit_article.html", form=form, article=art)

@bp.post("/article/<int:id>/delete")
@login_required
def delete_article(id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id:
        abort(403)
    db.session.delete(art)
    db.session.commit()
    flash("記事を削除しました", "success")
    return redirect(url_for(".log", username=current_user.username, site_id=art.site_id))

# app/routes.py

@bp.route("/<username>/articles/<int:id>/retry", methods=["POST"])
@login_required
def retry_article(username, id):
    art = Article.query.get_or_404(id)
    if art.user_id != current_user.id or username != current_user.username:
        abort(403)

    if not art.title_prompt or not art.body_prompt:
        flash("この記事は再生成できません（プロンプト未保存）", "error")
        return redirect(url_for("main.view_articles", username=username))

    art.status = "pending"
    art.progress = 0
    art.updated_at = datetime.utcnow()
    db.session.commit()

    # バックグラウンドで再生成
    from app.article_generator import _generate
    app = current_app._get_current_object()
    threading.Thread(
        target=_generate,
        args=(app, art.id, art.title_prompt, art.body_prompt),
        daemon=True
    ).start()

    flash("記事の再生成を開始しました。しばらくお待ちください。", "success")
    return redirect(url_for("main.view_articles", username=username))



@bp.post("/articles/bulk-delete")
@login_required
def bulk_delete_articles():
    ids = request.form.getlist("selected_ids")
    if not ids:
        flash("削除する記事が選択されていません", "warning")
        return redirect(request.referrer or url_for(".dashboard"))

    for aid in ids:
        article = Article.query.get(int(aid))
        if article and article.user_id == current_user.id:
            db.session.delete(article)

    db.session.commit()
    flash(f"{len(ids)}件の記事を削除しました", "success")
    return redirect(request.referrer or url_for(".dashboard"))



@bp.route("/debug-post/<int:aid>")
@login_required
def debug_post(aid):
    art = Article.query.get_or_404(aid)
    if art.user_id != current_user.id:
        abort(403)
    try:
        url = post_to_wp(art.site, art)
        return f"SUCCESS: {url}"
    except Exception as e:
        return f"ERROR: {e}", 500
    
import requests
from app.models import GSCAuthToken, db
import datetime

# Google OAuth2 設定
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")
GOOGLE_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


@bp.route("/authorize_gsc/<int:site_id>")
@login_required
def authorize_gsc(site_id):
    session["gsc_site_id"] = site_id  # 後でcallbackで参照するため保存
    auth_url = (
        f"{GOOGLE_AUTH_URL}?"
        f"response_type=code&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&scope={GOOGLE_SCOPE}&access_type=offline&prompt=consent"
    )
    return redirect(auth_url)


@bp.route("/oauth2callback")
@login_required
def oauth2callback():
    from app.models import Site

    code = request.args.get("code")
    if not code:
        flash("Google認証に失敗しました。", "danger")
        return redirect(url_for("main.gsc_connect"))

    site_id = session.get("gsc_site_id")
    site = Site.query.get_or_404(site_id)

    # トークン交換リクエスト
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    response = requests.post(GOOGLE_TOKEN_URL, data=data)
    if response.status_code != 200:
        flash("トークンの取得に失敗しました。", "danger")
        return redirect(url_for("main.gsc_connect"))

    tokens = response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 3600)
    expiry = datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)

    # 保存（存在する場合は更新）
    existing = GSCAuthToken.query.filter_by(site_id=site.id, user_id=current_user.id).first()
    if existing:
        existing.access_token = access_token
        existing.refresh_token = refresh_token
        existing.token_expiry = expiry
    else:
        new_token = GSCAuthToken(
            site_id=site.id,
            user_id=current_user.id,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expiry=expiry,
        )
        db.session.add(new_token)

    site.gsc_connected = True
    db.session.commit()

    flash(f"サイト「{site.name}」とGoogle Search Consoleの接続に成功しました。", "success")
    return redirect(url_for("main.gsc_connect"))

from app.forms import GenreForm

# ─────────── ジャンル管理ページ
@bp.route("/<username>/genres", methods=["GET", "POST"])
@login_required
def manage_genres(username):
    if current_user.username != username:
        abort(403)

    form = GenreForm()
    if form.validate_on_submit():
        # 🔹 既存ジャンル名と重複しないようにチェック（同一ユーザー内）
        existing = Genre.query.filter_by(user_id=current_user.id, name=form.name.data.strip()).first()
        if existing:
            flash("同じ名前のジャンルが既に存在します。", "warning")
        else:
            genre = Genre(
                name=form.name.data.strip(),
                description=form.description.data.strip(),
                user_id=current_user.id
            )
            db.session.add(genre)
            db.session.commit()
            flash("ジャンルを追加しました。", "success")
        return redirect(url_for("main.manage_genres", username=username))

    genres = Genre.query.filter_by(user_id=current_user.id).order_by(Genre.name).all()
    return render_template("genres.html", form=form, genres=genres)


# ─────────── ジャンル編集
@bp.route("/<username>/genres/edit/<int:genre_id>", methods=["GET", "POST"])
@login_required
def edit_genre(username, genre_id):
    if current_user.username != username:
        abort(403)

    genre = Genre.query.filter_by(id=genre_id, user_id=current_user.id).first_or_404()
    form = GenreForm(obj=genre)

    if form.validate_on_submit():
        genre.name = form.name.data.strip()
        genre.description = form.description.data.strip()
        db.session.commit()
        flash("ジャンルを更新しました。", "success")
        return redirect(url_for("main.manage_genres", username=username))

    return render_template("genres.html", form=form, genres=[], edit_genre=genre)


# ─────────── ジャンル削除
@bp.route("/<username>/genres/delete/<int:genre_id>", methods=["POST"])
@login_required
def delete_genre(username, genre_id):
    if current_user.username != username:
        abort(403)

    genre = Genre.query.filter_by(id=genre_id, user_id=current_user.id).first_or_404()
    db.session.delete(genre)
    db.session.commit()
    flash("ジャンルを削除しました。", "info")
    return redirect(url_for("main.manage_genres", username=username))


# -----------------------------------------------------------------
#────────── 外部SEO関連ルート ──────────
# -----------------------------------------------------------------

@bp.route("/external/sites")
@login_required
def external_seo_sites():
    from app.models import (
        Site, ExternalSEOJob, ExternalArticleSchedule,
        ExternalBlogAccount, BlogType, ExternalSEOJobLog, Article
    )
    from app import db
    from sqlalchemy.orm import selectinload
    from sqlalchemy import func, or_
    from datetime import datetime, timedelta, timezone

    sites = (
        Site.query
        .filter_by(user_id=current_user.id)
        .options(
            selectinload(Site.external_jobs),
            selectinload(Site.external_accounts),
        )
        .all()
    )

    job_map, key_set = {}, set()
    for s in sites:
        for job in s.external_jobs:
            if job.status == "archived":
                continue
            key = (s.id, job.blog_type)
            key_set.add(key)
            job_map[(s.id, job.blog_type.value.lower())] = job

    posted_counts = (
        db.session.query(
            ExternalBlogAccount.site_id,
            ExternalBlogAccount.blog_type,
            func.count(ExternalArticleSchedule.id),
        )
        .join(
            ExternalArticleSchedule,
            ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id,
        )
        .filter(
            ExternalArticleSchedule.status == "posted",
            ExternalBlogAccount.site_id.in_([sid for sid, _ in key_set]) if key_set else True,
            ExternalBlogAccount.blog_type.in_([bt for _, bt in key_set]) if key_set else True,
        )
        .group_by(ExternalBlogAccount.site_id, ExternalBlogAccount.blog_type)
        .all()
    )
    for site_id, blog_type, cnt in posted_counts:
        key = (site_id, blog_type.value.lower())
        if key in job_map:
            job_map[key].posted_cnt = cnt
    for job in job_map.values():
        if not hasattr(job, "posted_cnt"):
            job.posted_cnt = 0

    # メトリクス集計（アカウント単位）
    all_ld_account_ids = []
    for s in sites:
        for acc in (s.external_accounts or []):
            if acc.blog_type == BlogType.LIVEDOOR:
                all_ld_account_ids.append(acc.id)

    per_acc_total, per_acc_posted = {}, {}
    if all_ld_account_ids:
        for aid, cnt in (
            db.session.query(
                ExternalArticleSchedule.blog_account_id,
                func.count(ExternalArticleSchedule.id),
            )
            .filter(ExternalArticleSchedule.blog_account_id.in_(all_ld_account_ids))
            .group_by(ExternalArticleSchedule.blog_account_id)
            .all()
        ):
            per_acc_total[aid] = cnt

        for aid, cnt in (
            db.session.query(
                ExternalArticleSchedule.blog_account_id,
                func.count(ExternalArticleSchedule.id),
            )
            .filter(
                ExternalArticleSchedule.blog_account_id.in_(all_ld_account_ids),
                ExternalArticleSchedule.status == "posted",
            )
            .group_by(ExternalArticleSchedule.blog_account_id)
            .all()
        ):
            per_acc_posted[aid] = cnt

    for s in sites:
        livedoor_accounts = []
        for acc in (s.external_accounts or []):
            if acc.blog_type != BlogType.LIVEDOOR:
                continue

            setattr(acc, "captcha_done", bool(getattr(acc, "is_captcha_completed", False)))
            setattr(acc, "email_verified", bool(getattr(acc, "is_email_verified", False)))

            livedoor_blog_id = getattr(acc, "livedoor_blog_id", None)
            setattr(acc, "blog_created", bool(livedoor_blog_id))
            setattr(acc, "api_key", getattr(acc, "atompub_key_enc", None))

            title = (
                getattr(acc, "nickname", None)
                or getattr(acc, "username", None)
                or livedoor_blog_id
                or f"account#{acc.id}"
            )
            setattr(acc, "blog_title", title)

            public_url = getattr(acc, "public_url", None)
            if not public_url and livedoor_blog_id:
                public_url = f"https://{livedoor_blog_id}.livedoor.blog/"
            setattr(acc, "public_url", public_url)

            total  = per_acc_total.get(acc.id, 0)
            posted = per_acc_posted.get(acc.id, 0)
            generated = max(total - posted, 0)
            setattr(acc, "stat_total", total)
            setattr(acc, "stat_posted", posted)
            setattr(acc, "stat_generated", generated)
            setattr(acc, "has_activity", total > 0)

            livedoor_accounts.append(acc)

        # ▼ livedoor_blog_id が同じものは 1 件に統合（NULLは統合しない）
        dedup_map = {}
        for acc in livedoor_accounts:
            key = getattr(acc, "livedoor_blog_id", None)
            if key is None:
                # blog_id 未確定はそのまま別カードとして扱う
                dedup_map[f"__id__:{acc.id}"] = acc
                continue

            prev = dedup_map.get(key)
            if not prev:
                dedup_map[key] = acc
                continue

            # どちらを残すか：APIキー > CAPTCHA済み > id新しい
            def score(a):
                return (
                    2 if getattr(a, "api_key", None) else
                    1 if getattr(a, "captcha_done", False) else
                    0,
                    getattr(a, "id", 0)
                )

            if score(acc) > score(prev):
                dedup_map[key] = acc

        dedup_list = list(dedup_map.values())
        s.livedoor_accounts = dedup_list
        s.ld_count = len(dedup_list)  # ← この値をテンプレの (n) に使う

    # ▼ 各サイトの「通常記事（外部SEO以外で投稿済み）」件数をテンプレに渡す
    site_ids = [s.id for s in sites]
    if site_ids:
        normal_counts = dict(
            db.session.query(Article.site_id, func.count(Article.id))
            .filter(Article.site_id.in_(site_ids))
            .filter(or_(Article.source.is_(None), Article.source != "external"))
            .filter(Article.status.in_(["posted", "published"]))  # ← done を除外（WP投稿済みのみ）
            .group_by(Article.site_id)
            .all()
        )
    else:
        normal_counts = {}
    for s in sites:
        # テンプレート側で can_start_extseo 判定用に参照
        s.normal_post_count = normal_counts.get(s.id, 0)

    # === GSC 直近28日合計（JSTで「昨日まで」）→ サイト一覧と完全同一ロジック ===
    from app.models import GSCDailyTotal  # ← サイト一覧と同じモデルを使用

    clicks28 = {}
    impr28   = {}
    if site_ids:
        start_d, end_d = _gsc_window_by_latest_db(28)

        rows = (
            db.session.query(
                GSCDailyTotal.site_id,
                func.coalesce(func.sum(GSCDailyTotal.clicks), 0).label("clicks"),
                func.coalesce(func.sum(GSCDailyTotal.impressions), 0).label("impressions"),
            )
            .filter(GSCDailyTotal.site_id.in_(site_ids))
            .filter(GSCDailyTotal.date >= start_d, GSCDailyTotal.date <= end_d)
            .group_by(GSCDailyTotal.site_id)
            .all()
        )
        clicks28 = {sid: c for sid, c, _ in rows}
        impr28   = {sid: i for sid, _, i in rows}

    for s in sites:
        s.clicks_28d      = clicks28.get(s.id, 0)
        s.impressions_28d = impr28.get(s.id, 0)

    return render_template(
        "external_sites.html",
        sites=sites,
        job_map=job_map,
        ExternalSEOJobLog=ExternalSEOJobLog,
    )




@bp.post("/external/start")
@login_required
def start_external_seo() -> "Response":
    """
    HTMX から送られてくる

        site_id=<数字>&blog=<文字列>

    を受け取り、GPTベースのAIエージェントでアカウント作成を即時実行する。
    - blog=note → run_note_signup()
    - blog=hatena → run_hatena_signup()
    - blog=livedoor → run_livedoor_signup()
    """
    from flask import request, abort, jsonify, render_template
    from app.models import Site
    from app.enums import BlogType  # BlogType Enum
    from app.services.blog_signup import (
        note_signup,
        hatena_signup,
        livedoor_signup,
    )

    site_id = request.form.get("site_id", type=int)
    blog = (request.form.get("blog") or "").lower()

    if not site_id or not blog:
        return "site_id と blog は必須です", 400

    # BlogType Enum変換（存在しないblogなら400）
    try:
        blog_type = BlogType(blog)
    except ValueError:
        return "不正なブログタイプ", 400

    # サイト取得と所有権チェック（管理者はスキップ）
    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    # --- 🎯 GPTエージェントの実行 ---
    try:
        if blog_type == BlogType.NOTE:
            note_signup.signup(site)
        elif blog_type == BlogType.HATENA:
            hatena_signup.signup(site)
        elif blog_type == BlogType.LIVEDOOR:
            livedoor_signup.signup(site)
        else:
            return f"未対応のブログ: {blog}", 400
    except Exception as e:
        return f"AIエージェント失敗: {str(e)}", 500

    # HTMX対応
    if request.headers.get("HX-Request"):
        return render_template(
            "_job_progress.html",
            site_id=site_id,
            blog=blog_type.value,
            job=None
        )
    return jsonify(status="success")



# -----------------------------------------------------------------
# 外部SEO: 進捗パネル HTMX 用
# -----------------------------------------------------------------
@bp.route("/external/jobs/<int:site_id>")
@login_required
def external_seo_job_status(site_id):
    from app.models import ExternalSEOJob

    job = (ExternalSEOJob.query
           .filter_by(site_id=site_id)
           .order_by(ExternalSEOJob.id.desc())
           .first())

    return render_template("_job_progress.html",
                           job=job,
                           site_id=site_id)

# ──────────────────────────────────────────
# 外部SEO: 投稿スケジュール一覧表示
# ──────────────────────────────────────────
@bp.route("/external/schedules/<int:site_id>")
@login_required
def external_schedules(site_id):
    from app.models import ExternalArticleSchedule, Keyword, ExternalBlogAccount

    # blog_account_id を site_id で絞る
    schedules = (
        db.session.query(ExternalArticleSchedule, Keyword, ExternalBlogAccount)
        .join(Keyword, ExternalArticleSchedule.keyword_id == Keyword.id)
        .join(ExternalBlogAccount, ExternalArticleSchedule.blog_account_id == ExternalBlogAccount.id)
        .filter(ExternalBlogAccount.site_id == site_id)
        .order_by(ExternalArticleSchedule.scheduled_date.asc())
        .all()
    )
    return render_template("external_schedules.html",
                           schedules=schedules,
                           site_id=site_id)

from flask import send_file, make_response
from .services.blog_signup.crypto_utils import decrypt
from app.models import ExternalBlogAccount, BlogType
import asyncio, json, time


# -----------------------------------------------------------
# ユーザー向け: 自分の外部ブログアカウント一覧（検索・絞込・ソート対応）
# -----------------------------------------------------------

@bp.route("/external/accounts")
@login_required
def external_accounts():
    from app.models import ExternalBlogAccount, Site, ExternalArticleSchedule, BlogType
    from app.services.blog_signup.crypto_utils import decrypt
    from sqlalchemy import or_, func, case
    from sqlalchemy.orm import aliased

    blog_type = request.args.get("blog_type")
    sort      = request.args.get("sort")
    search    = request.args.get("q", "").strip()
    site_id   = request.args.get("site_id", type=int)

    # ベース: ログインユーザーのサイトに属する（site_id が NULL でも可）
    base = (
        db.session.query(ExternalBlogAccount.id)
        .outerjoin(Site, ExternalBlogAccount.site_id == Site.id)
        .filter(
            (ExternalBlogAccount.site_id == None) |  # noqa: E711
            (Site.user_id == current_user.id)
        )
    )
    if site_id:
        base = base.filter(ExternalBlogAccount.site_id == site_id)
    if blog_type:
        # Enum の可能性に配慮（文字列でも Enum でも比較できるように）
        try:
            bt = BlogType(blog_type)  # 文字列→Enum
            base = base.filter(ExternalBlogAccount.blog_type == bt)
        except Exception:
            base = base.filter(ExternalBlogAccount.blog_type == blog_type)

    if search:
        base = base.filter(or_(
            ExternalBlogAccount.email.ilike(f"%{search}%"),
            ExternalBlogAccount.nickname.ilike(f"%{search}%"),
            ExternalBlogAccount.username.ilike(f"%{search}%"),
        ))

    # 集計に使う別名（※ JOIN は schedule のみ。Keyword/Article には JOIN しない）
    S = aliased(ExternalArticleSchedule)

    # 各アカウント行ごとの集計（1アカウント=1行）
    # - total_cnt    : COUNT(DISTINCT S.id)
    # - posted_cnt   : SUM(CASE WHEN S.status='posted' THEN 1 ELSE 0 END)
    # - generated_cnt: SUM(CASE WHEN S.article_id IS NOT NULL THEN 1 ELSE 0 END)
    per_acc_rows = (
        db.session.query(
            ExternalBlogAccount,
            func.count(func.distinct(S.id)).label("total_cnt"),
            func.sum(case((S.status == "posted", 1), else_=0)).label("posted_cnt"),
            func.sum(case((S.article_id != None, 1), else_=0)).label("generated_cnt")  # noqa: E711
        )
        .select_from(ExternalBlogAccount)
        .outerjoin(S, S.blog_account_id == ExternalBlogAccount.id)
        .outerjoin(Site, ExternalBlogAccount.site_id == Site.id)
        .filter(base.whereclause)  # ベースのフィルタを適用
        .group_by(ExternalBlogAccount.id)
        .all()
    )

    # （blog_type, blog_id）でユニーク化し、代表1件に集計を合算
    def score(acc):
        # 代表選定優先度: APIキー > CAPTCHA済み > id
        return (
            2 if getattr(acc, "atompub_key_enc", None) else
            1 if getattr(acc, "is_captcha_completed", False) else
            0,
            getattr(acc, "id", 0)
        )

    groups = {}  # key -> {"repr": acc, "total":int, "posted":int, "generated":int, "raw":[acc,...]}
    for acc, total_cnt, posted_cnt, generated_cnt in per_acc_rows:
        key_blog_id = acc.livedoor_blog_id or f"__id__:{acc.id}"
        key = (acc.blog_type, key_blog_id)
        g = groups.get(key)
        total_i     = int(total_cnt or 0)
        posted_i    = int(posted_cnt or 0)
        generated_i = int(generated_cnt or 0)

        if not g:
            groups[key] = {
                "repr": acc,
                "total": total_i,
                "posted": posted_i,
                "generated": generated_i,
                "raw": [acc],
            }
        else:
            # 代表を差し替える場合がある
            if score(acc) > score(g["repr"]):
                g["repr"] = acc
            # 集計は合算
            g["total"]     += total_i
            g["posted"]    += posted_i
            g["generated"] += generated_i
            g["raw"].append(acc)

    # 表示用リスト（代表 acc に合算済みの数値を持たせる）
    accts = []
    for _, g in groups.items():
        a = g["repr"]
        a.total_cnt     = g["total"]
        a.posted_cnt    = g["posted"]
        a.generated_cnt = g["generated"]
        a._raw_count    = len(g["raw"])  # 任意：統合件数（表示したければテンプレで参照）
        accts.append(a)

    # 並び替え（ユニーク化後の値で）
    def sort_key(a):
        if sort == "posted_asc":
            return (a.posted_cnt or 0, a.id)
        if sort == "posted_desc":
            return (-(a.posted_cnt or 0), a.id)
        if sort == "generated_asc":
            return (a.generated_cnt or 0, a.id)
        if sort == "generated_desc":
            return (-(a.generated_cnt or 0), a.id)
        if sort == "total_asc":
            return (a.total_cnt or 0, a.id)
        # default: total_desc
        return (-(a.total_cnt or 0), a.id)

    accts.sort(key=sort_key)

    all_sites = Site.query.filter_by(user_id=current_user.id).all()

    return render_template(
        "external_accounts.html",
        accts=accts,                 # ← ユニーク化後の代表たち
        all_sites=all_sites,
        decrypt=decrypt,
        site_id=site_id,
        selected_blog_type=blog_type,
        selected_sort=sort,
        search_query=search
    )




@bp.route("/external/account/<int:acct_id>/articles")
@login_required
def external_account_articles(acct_id):
    from app.models import ExternalBlogAccount, ExternalArticleSchedule, Keyword, Article

    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    site = acct.site
    if site.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    rows = (
        db.session.query(ExternalArticleSchedule, Keyword, Article)
        .join(Keyword, ExternalArticleSchedule.keyword_id == Keyword.id)
        .outerjoin(Article, Article.id == ExternalArticleSchedule.article_id)
        .filter(ExternalArticleSchedule.blog_account_id == acct_id)
        # ▼ 修正：古い順（ASC）＋ タイブレークに schedule.id
        .order_by(ExternalArticleSchedule.scheduled_date.asc(),
                  ExternalArticleSchedule.id.asc())
        .all()
    )

    return render_template(
        "external_articles.html",
        acct=acct, site=site, rows=rows
    )


@bp.route("/external/article/<int:article_id>/preview")
@login_required
def external_article_preview(article_id):
    from app.models import Article

    art = Article.query.get_or_404(article_id)

    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    return render_template("external_article_preview.html", article=art)



# 外部SEO記事 編集
@bp.route("/external/article/<int:article_id>/edit", methods=["GET", "POST"])
@login_required
def external_article_edit(article_id):
    from app.models import Article
    art = Article.query.get_or_404(article_id)

    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    if request.method == "POST":
        art.title = request.form.get("title", art.title)
        art.body = request.form.get("body", art.body)
        db.session.commit()
        flash("記事を更新しました", "success")
        # 確実に戻れるように
        return redirect(request.referrer or url_for("main.external_schedules", site_id=art.site_id))

    return render_template("external_article_edit.html", article=art)

# 外部SEO記事 削除
@bp.route("/external/article/<int:article_id>/delete", methods=["POST"])
@login_required
def external_article_delete(article_id):
    from app.models import Article, ExternalArticleSchedule, Keyword

    art = Article.query.get_or_404(article_id)
    if art.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    # Article から Keyword.id を引く
    kw = Keyword.query.filter_by(site_id=art.site_id, keyword=art.keyword).first()

    if kw:
        schedules = ExternalArticleSchedule.query.filter_by(keyword_id=kw.id).all()
        for sched in schedules:
            db.session.delete(sched)

    db.session.delete(art)
    db.session.commit()
    flash("記事を削除しました", "success")
    # 元画面に戻す（acct_id が取れないので referrer 優先）
    return redirect(request.referrer or url_for("main.external_schedules", site_id=art.site_id))


# 外部SEO記事 即時投稿
@bp.route("/external/schedule/<int:schedule_id>/post_now", methods=["POST"])
@login_required
def external_schedule_post_now(schedule_id):
    from datetime import datetime
    from flask import current_app, request, redirect, url_for, flash, abort
    from flask_login import current_user
    from app import db
    from app.models import ExternalArticleSchedule
    from app.tasks import _run_external_post_job  # ← ここを修正

    sched = ExternalArticleSchedule.query.get_or_404(schedule_id)
    acct = sched.blog_account
    site = acct.site

    # 所有権チェック
    if site.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    # 直ちに実行対象へ（UTC naive）
    sched.scheduled_date = datetime.utcnow()
    sched.status = "pending"
    db.session.commit()

    try:
        # pending を処理
        _run_external_post_job(current_app._get_current_object(), schedule_id=schedule_id)
        flash("即時投稿を開始しました。しばらくしてページを更新してください。", "success")
    except Exception as e:
        current_app.logger.exception("[external] post_now failed")
        flash(f"即時投稿に失敗しました: {e}", "danger")

    return redirect(request.referrer or url_for("main.external_account_articles", acct_id=acct.id))

# --- 一括削除: 外部ブログアカウント + 予約 +（安全条件下の）生成記事 ---
@bp.post("/external/account/<int:acct_id>/delete")
@login_required
def external_account_delete(acct_id):
    from app.models import (
        ExternalBlogAccount, ExternalArticleSchedule, Keyword, Article, Site
    )
    from sqlalchemy import exists, and_, select
    from app import db

    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    site: Site = acct.site

    # 権限
    if not current_user.is_admin and site.user_id != current_user.id:
        return {"ok": False, "error": "権限がありません"}, 403

    # まず、このアカウントの全スケジュールを取得（Keywordも使うためIDを保持）
    schedules = (
        db.session.query(ExternalArticleSchedule)
        .filter(ExternalArticleSchedule.blog_account_id == acct.id)
        .all()
    )
    keyword_ids = [s.keyword_id for s in schedules if getattr(s, "keyword_id", None)]
    # ID→テキストを得る（Articleは keyword(テキスト) 基準で紐付けられているため）
    kw_texts = []
    if keyword_ids:
        kw_texts = [
            k.keyword for k in db.session.query(Keyword).filter(Keyword.id.in_(keyword_ids)).all()
        ]

    # このアカウント以外でも同じキーワードIDが使われているか（残す条件）
    # → Articleは「同じ keyword テキスト」を共有し得るので、
    #   “他アカウントの予約が同一KeywordIDを参照していない”記事のみ削除対象とする
    if kw_texts:
        # schedules テーブルで “同一 keyword_id かつ 別アカウント” が存在しないことを条件に Article を削除
        # Article は site_id と source='external' で限定
        subq_other = (
            db.session.query(ExternalArticleSchedule.id)
            .filter(
                ExternalArticleSchedule.keyword_id.in_(keyword_ids),
                ExternalArticleSchedule.blog_account_id != acct.id
            )
            .exists()
        )
        # 削除対象 Article の選別
        articles_q = (
            db.session.query(Article)
            .filter(
                Article.site_id == site.id,
                Article.source == "external",
                Article.keyword.in_(kw_texts),
                ~subq_other   # 他アカウントの予約が無いキーワードのみ
            )
        )
        deleted_articles = articles_q.delete(synchronize_session=False)
    else:
        deleted_articles = 0

    # スケジュール削除
    db.session.query(ExternalArticleSchedule)\
        .filter(ExternalArticleSchedule.blog_account_id == acct.id)\
        .delete(synchronize_session=False)

    # アカウント削除
    db.session.delete(acct)
    db.session.commit()

    return {"ok": True, "deleted_articles": int(deleted_articles)}


# -----------------------------------------------------------------
# Livedoor 手動保存: APIキー/エンドポイント
# -----------------------------------------------------------------
@bp.post("/external/livedoor/credentials/save")
@login_required
def livedoor_credentials_save():
    """
    入力: site_id, account_id, blog_id, endpoint, api_key
    機能: 検証・正規化して保存（DB優先 / 暫定JSONフォールバック）
    戻り: { ok: true, masked_key: "••••abcd", status: "unknown" } or { ok:false, error:"..." }
    """
    from flask import request, jsonify, abort
    from app import db
    from app.models import Site, ExternalBlogAccount, BlogType
    import re, urllib.parse
    # 暫定フォールバック（DB未対応環境）: JSON保存関数
    try:
        from app.services.blog_signup.livedoor_signup import save_livedoor_credentials as _json_save
    except Exception:
        _json_save = None

    def _mask_tail(s: str, n: int = 4) -> str:
        if not s:
            return ""
        tail = s[-n:] if len(s) >= n else s
        return "••••" + tail

    def _normalize_endpoint(raw: str) -> str:
        v = (raw or "").strip()
        if not v:
            return v
        # スキーム付与
        if not re.match(r"^https?://", v, re.I):
            v = "https://" + v
        # 余計な空白や連続スラッシュの整理（プロトコル部は除く）
        parts = urllib.parse.urlsplit(v)
        path = re.sub(r"/{2,}", "/", parts.path or "/")
        # /atompub が含まれていなければ付与（末尾スラッシュは1つに）
        if not re.search(r"/atompub/?$", path, re.I):
            path = path.rstrip("/") + "/atompub"
        path = path.rstrip("/")  # 最終的に末尾スラなしに統一
        v2 = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
        return v2

    def _validate_blog_id(bid: str) -> bool:
        return bool(re.match(r"^[a-z0-9_]{3,20}$", (bid or "").strip()))

    # --- 入力取得（JSON or form） ---
    getv = (request.get_json(silent=True) or request.form)
    site_id    = getv.get("site_id", type=int)
    account_id = getv.get("account_id", type=int)
    blog_id    = (getv.get("blog_id") or "").strip()
    endpoint   = (getv.get("endpoint") or "").strip()
    api_key    = (getv.get("api_key") or "").strip()

    if not site_id or not account_id:
        return jsonify(ok=False, error="site_id と account_id は必須です"), 400
    if not blog_id or not endpoint or not api_key:
        return jsonify(ok=False, error="blog_id / endpoint / api_key は必須です"), 400
    if not _validate_blog_id(blog_id):
        return jsonify(ok=False, error="blog_id の形式が不正です（半角英数+_ 3〜20 文字）"), 400

    # 所有権
    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)
    acct = ExternalBlogAccount.query.get_or_404(account_id)
    if acct.site_id != site.id and (not current_user.is_admin):
        return jsonify(ok=False, error="アカウントがサイトに属していません"), 400
    # Livedoor 以外は拒否
    if getattr(acct, "blog_type", None) != BlogType.LIVEDOOR:
        return jsonify(ok=False, error="このアカウントは Livedoor ではありません"), 400

    # 正規化
    endpoint_norm = _normalize_endpoint(endpoint)
    if not re.match(r"^https://[^/]+/.*", endpoint_norm, re.I):
        return jsonify(ok=False, error="endpoint URL が不正です"), 400

    # --- 保存（DB優先 / フォールバックJSON） ---
    saved = False
    try:
        # できるだけ広くフィールドに対応（環境差異を吸収）
        if hasattr(acct, "livedoor_blog_id"):
            acct.livedoor_blog_id = blog_id
        if hasattr(acct, "atompub_endpoint"):
            acct.atompub_endpoint = endpoint_norm
        if hasattr(acct, "atompub_key_enc"):
            # 暗号化フィールド想定
            acct.atompub_key_enc = api_key
        elif hasattr(acct, "api_key"):
            # 平文フィールドがある環境向け
            acct.api_key = api_key
        # 未テスト状態に戻す（Boolean/Nullable 両対応）
        if hasattr(acct, "api_post_enabled"):
            try:
                acct.api_post_enabled = None
            except Exception:
                pass
        db.session.commit()
        saved = True
    except Exception:
        db.session.rollback()
        saved = False

    # DBが使えない（または失敗）環境では暫定JSONに保存
    if not saved:
        if _json_save is None:
            return jsonify(ok=False, error="保存に失敗しました（DB/JSONともに不可）"), 500
        try:
            _json_save(
                site_id=site_id,
                account_id=account_id,
                livedoor_blog_id=blog_id,
                endpoint=endpoint_norm,
                api_key=api_key,
            )
        except Exception as e:
            return jsonify(ok=False, error=f"保存に失敗しました: {e}"), 500

    return jsonify(ok=True, masked_key=_mask_tail(api_key, 4), status="unknown")


# -----------------------------------------------------------------
# Livedoor 接続テスト（軽量 AtomPub GET）
# -----------------------------------------------------------------
@bp.post("/external/livedoor/credentials/test")
@login_required
def livedoor_credentials_test():
    """
    入力: site_id, account_id
    機能: 保存済み blog_id / endpoint / api_key で軽量接続確認
    戻り: { ok:true } もしくは { ok:false, detail:"..." }
    副作用: ExternalBlogAccount.api_post_enabled を True/False に更新
    """
    from flask import request, jsonify, abort
    from app import db
    from app.models import Site, ExternalBlogAccount, BlogType
    import requests

    # livedoor_atompub 側に probe 関数があれば利用、無ければフォールバック
    try:
        from app.services.livedoor_atompub import probe_auth as _probe_auth
    except Exception:
        _probe_auth = None

    getv = (request.get_json(silent=True) or request.form)
    site_id    = getv.get("site_id", type=int)
    account_id = getv.get("account_id", type=int)
    if not site_id or not account_id:
        return jsonify(ok=False, detail="site_id と account_id は必須です"), 400

    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)
    acct = ExternalBlogAccount.query.get_or_404(account_id)
    if acct.site_id != site.id and (not current_user.is_admin):
        return jsonify(ok=False, detail="アカウントがサイトに属していません"), 400
    if getattr(acct, "blog_type", None) != BlogType.LIVEDOOR:
        return jsonify(ok=False, detail="このアカウントは Livedoor ではありません"), 400

    blog_id  = getattr(acct, "livedoor_blog_id", None)
    endpoint = getattr(acct, "atompub_endpoint", None)
    # キーは環境によりフィールド名が異なりうる
    api_key  = getattr(acct, "atompub_key_enc", None) or getattr(acct, "api_key", None)
    if not (blog_id and endpoint and api_key):
        return jsonify(ok=False, detail="設定が不足しています（blog_id / endpoint / api_key）"), 400

    ok = False
    detail = ""
    try:
        if callable(_probe_auth):
            # 既存ユーティリティがある場合はそれを最優先
            # 期待: 戻り値 True/False、例外でエラー詳細
            ok = bool(_probe_auth(endpoint=endpoint, api_key=api_key, blog_id=blog_id))
            detail = "" if ok else "認証エラー"
        else:
            # フォールバック: 最小の GET を投げ、200/401/403 で判定（超軽量）
            # 認証ヘッダ方式が環境依存のため、ここでは疎通/認証失敗の大枠のみを判定
            try:
                resp = requests.get(endpoint, timeout=6)
                if resp.status_code // 100 == 2:
                    ok = True
                elif resp.status_code in (401, 403):
                    ok = False
                    detail = "認証エラー"
                else:
                    ok = False
                    detail = f"HTTP {resp.status_code}"
            except requests.Timeout:
                ok = False
                detail = "タイムアウト"
    except Exception as e:
        ok = False
        detail = f"接続エラー: {e}"

    # フラグ更新（Nullable/Boolean を許容）
    try:
        if hasattr(acct, "api_post_enabled"):
            acct.api_post_enabled = True if ok else False
        db.session.commit()
    except Exception:
        db.session.rollback()

    if ok:
        return jsonify(ok=True)
    return jsonify(ok=False, detail=(detail or "接続に失敗しました")), 200


# -----------------------------------------------------------
# 管理者向け: 全ユーザーの外部ブログアカウント一覧
# -----------------------------------------------------------
@admin_bp.route("/admin/blog_accounts")
@login_required
def admin_blog_accounts():
    if not current_user.is_admin:
        abort(403)

    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import decrypt

    accts = (ExternalBlogAccount
             .query.order_by(ExternalBlogAccount.created_at.desc())
             .all())

    # ★ パスを "admin/xxx.html" に変更
    return render_template(
        "admin/admin_blog_accounts.html",
        accts    = accts,
        decrypt  = decrypt,
    )


# ---------------------------------------------------------
# 🔐 管理者専用：ワンクリックで対象ブログへログインする中間ページ
# ---------------------------------------------------------
# app/routes.py など
from flask import Blueprint, request, abort, render_template_string
from flask_login import login_required, current_user
from app import db


# ---------------------------------------------------------
# 🔐 管理者専用：ワンクリック自動ログイン
# ---------------------------------------------------------
@admin_bp.route("/admin/blog_login", methods=["POST"])
@login_required
def admin_blog_login():
    """
    管理者が「ワンクリックログイン」を押した時に呼び出される。
    - 対応サービス (note / hatena …) は自動 POST
    - 未対応サービスは資格情報を表示
    """
    if not current_user.is_admin:
        abort(403)

    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import decrypt

    acct_id = request.form.get("account_id", type=int)
    if not acct_id:
        abort(400, "account_id missing")

    acct: ExternalBlogAccount | None = ExternalBlogAccount.query.get(acct_id)
    if not acct:
        abort(404, "account not found")

    email    = decrypt(acct.email)
    password = decrypt(acct.password)
    username = acct.username

    login_map = {
        "note": {
            "url": "https://note.com/login",
            "user_field": "email",
            "pass_field": "password",
        },
        "hatena": {
            "url": "https://www.hatena.ne.jp/login",
            "user_field": "name",
            "pass_field": "password",
        },
        # ここに他ブログを追加
    }

    cfg = login_map.get(acct.blog_type.value)

    # --- 対応ブログ：自動 POST フォーム ----
    if cfg:
        return f"""
        <!doctype html><html lang="ja"><head><meta charset="utf-8">
        <title>auto-login</title></head><body>
          <p style="font-family:sans-serif;margin-top:2rem">
            {acct.blog_type.value} にリダイレクト中…
          </p>
          <form id="f" action="{cfg['url']}" method="post">
            <input type="hidden" name="{cfg['user_field']}" value="{email}">
            <input type="hidden" name="{cfg['pass_field']}" value="{password}">
          </form>
          <script>setTimeout(()=>document.getElementById('f').submit(), 300);</script>
        </body></html>
        """

    # --- 未対応ブログ：資格情報表示 ----
    return render_template_string("""
      <!doctype html><html lang="ja"><head><meta charset="utf-8">
      <title>資格情報</title></head><body style="font-family:sans-serif">
        <h2>手動ログインが必要です</h2>
        <ul>
          <li><b>サービス</b>: {{ blog }}</li>
          <li><b>ユーザー名</b>: {{ uname }}</li>
          <li><b>メール</b>: {{ mail }}</li>
          <li><b>パスワード</b>: {{ pwd }}</li>
        </ul>
      </body></html>
    """, blog=acct.blog_type.value, uname=username, mail=email, pwd=password)



# -----------------------------------------------------------
# ワンクリックログイン 
# -----------------------------------------------------------
@bp.route("/external/login/<int:acct_id>")
@login_required
def blog_one_click_login(acct_id):
    acct = ExternalBlogAccount.query.get_or_404(acct_id)
    if not (current_user.is_admin or acct.site.user_id == current_user.id):
        abort(403)

    from app.services.blog_signup.crypto_utils import decrypt

    if acct.blog_type == BlogType.NOTE:
        from app.services.blog_signup.note_login import get_note_cookies
        cookies = asyncio.run(get_note_cookies(decrypt(acct.email), decrypt(acct.password)))
        resp = make_response(redirect("https://note.com"))
        for c in cookies:
            resp.set_cookie(
                key=c["name"], value=c["value"],
                domain=".note.com", path="/", secure=True, httponly=True,
                samesite="Lax", expires=int(time.time()) + 60*60
            )
        return resp

    elif acct.blog_type == BlogType.LIVEDOOR:
        # ★ 追加：Livedoor対応
        from app.services.blog_signup.livedoor_login import get_livedoor_cookies
        cookies = asyncio.run(get_livedoor_cookies(decrypt(acct.email), decrypt(acct.password)))
        # 管理画面側に入れる
        resp = make_response(redirect("https://livedoor.blogcms.jp/member/"))
        for c in cookies:
            resp.set_cookie(
                key=c["name"], value=c["value"],
                domain=c.get("domain", ".livedoor.com"),
                path="/", secure=True, httponly=True,
                samesite="Lax", expires=int(time.time()) + 60*60
            )
        return resp

    else:
        abort(400, "Login not supported yet")



# ====== ルート先頭の import 付近に追記/置換 ======
from flask import session as flask_session  # 既にあればOK
from app.services.blog_signup.livedoor_signup import (
    generate_safe_id, generate_safe_password,
    prepare_captcha as ld_prepare_captcha,   # 新API名
    submit_captcha as ld_submit_captcha,     # 新API名
    suggest_livedoor_blog_id,
    poll_latest_link_gw,                     # メール認証リンク取得
    generate_livedoor_id_candidates,         # ★ 追加：実際に利用しているため
)
from app.services.mail_utils.mail_gw import create_inbox
from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
from app.services.pw_controller import pwctl  # セッションの明示クローズ用
# 既存 import 群の近くに追記
from flask import current_app  # submit_captcha で使っているため
from app.services.pw_session_store import (
    save as pw_save,
    get_cred as pw_get,
    clear as pw_clear,
)
# ※ pw_set は存在しないため import しない（ImportError対策）

# ====== /prepare_captcha ======
@bp.route("/prepare_captcha", methods=["POST"])
@login_required
def prepare_captcha():
    from app.models import Site, ExternalBlogAccount
    from app import db
    from flask import jsonify, request, url_for
    from uuid import uuid4
    from pathlib import Path
    import time as _time
    import logging
    logger = logging.getLogger(__name__)

    # ---------- ✅ 追加: DBコミットを堅牢化 ----------
    def safe_commit(session, retries: int = 1) -> bool:
        """
        OperationalError(切断など)時に rollback→最大1回だけ再実行。
        失敗したら False。その他の例外は上位へ投げない（呼び出し側で握りつぶす方針）。
        """
        try:
            session.commit()
            return True
        except OperationalError as e:
            try:
                current_app.logger.warning("safe_commit: OperationalError on commit, retrying once: %s", e)
                session.rollback()
                session.commit()
                return True
            except OperationalError as e2:
                current_app.logger.exception("safe_commit: retry failed: %s", e2)
                session.rollback()
                return False
        except Exception:
            # ここでは堅く失敗に倒す（呼び出し側で握りつぶす設計）
            current_app.logger.exception("safe_commit: non-OperationalError on commit")
            session.rollback()
            return False

    site_id    = request.form.get("site_id", type=int)
    blog       = request.form.get("blog")  # "livedoor"
    account_id = request.form.get("account_id", type=int)

    if not site_id or not blog:
        return jsonify({"captcha_url": None, "error": "site_id または blog が指定されていません",
                        "site_id": site_id, "account_id": account_id})

    site = Site.query.get(site_id)
    if not site or (not current_user.is_admin and site.user_id != current_user.id):
        return jsonify({"captcha_url": None, "error": "権限がありません",
                        "site_id": site_id, "account_id": account_id})

    # 所有アカウント検証（任意）
    acct = ExternalBlogAccount.query.get(account_id) if account_id else None
    if acct:
        if acct.site_id != site_id:
            return jsonify({"captcha_url": None, "error": "account_id が site_id に属していません",
                            "site_id": site_id, "account_id": account_id})
        if (not current_user.is_admin) and (acct.site.user_id != current_user.id):
            return jsonify({"captcha_url": None, "error": "権限がありません",
                            "site_id": site_id, "account_id": account_id})

    # 仮登録データ（メール & 候補 blog_id）
    email, token = create_inbox()  # 既存
    if not (email and token):
        return jsonify({"ok": False, "error": "mailbox_init_failed"}), 500

    desired_blog_id = request.form.get("blog_id") or request.form.get("sub") or None
    # 英字開始・3–20・英数＋_ 準拠の候補ロジックを採用
    livedoor_id  = generate_livedoor_id_candidates(site)[0]
    password     = generate_safe_password()

    try:
        base_text = site.name or site.url or ""
        desired_blog_id = suggest_livedoor_blog_id(base_text, db.session)
    except Exception:
        desired_blog_id = None

    # ▶ 新API: Playwright セッションを作って CAPTCHA 画像を保存
    try:
        session_id, img_abs_path = ld_prepare_captcha(email, livedoor_id, password)
    except Exception:
        logger.exception("[prepare_captcha] CAPTCHA生成で例外が発生")
        return jsonify({"captcha_url": None, "error": "CAPTCHAの準備に失敗しました",
                        "site_id": site_id, "account_id": account_id})
    
    # ★ 追加：資格情報を sid 単位で保存（フォームや Flask セッションに依存しない）
    pw_save(session_id,
            email=email,
            password=password,
            livedoor_id=livedoor_id,
            token=token,
            site_id=site_id,
            account_id=account_id,
            desired_blog_id=desired_blog_id)
    
    # 追加保存は不要。/submit_captcha は pw_session_store.pw_get(session_id) を参照する前提

    # 画像URL化
    img_name = Path(img_abs_path).name
    ts = int(_time.time())
    captcha_url = url_for("static", filename=f"captchas/{img_name}", _external=True) + f"?v={ts}"

    # セッション保持（次の /submit_captcha 用）
    flask_session.update({
        "captcha_email": email,
        "captcha_nickname": livedoor_id,
        "captcha_password": password,
        "captcha_token": token,
        "captcha_site_id": site_id,
        "captcha_blog": blog,
        "captcha_image_filename": img_name,
        "captcha_session_id": session_id,
        "captcha_account_id": account_id,
        "captcha_desired_blog_id": desired_blog_id,
    })

    # 任意：DBログ
    if acct:
        acct.captcha_session_id = session_id
        acct.captcha_image_path = f"captchas/{img_name}"
        # ✅ DB書き込みは“任意”。失敗してもUIは session_id で回復できるため成功返却を優先
        if not safe_commit(db.session, retries=1):
            current_app.logger.warning(
                "[prepare_captcha] DB更新(外部アカウントへのcaptcha_session_id設定)に失敗しましたが続行します "
                "(site_id=%s account_id=%s session_id=%s)", site_id, account_id, session_id
            )

    return jsonify({
        "ok": True,
        "captcha_url": captcha_url,
        "site_id": site_id,
        "account_id": account_id,
        # ← ★ これを必ず返す（フロントが submit 時に同封）
        "session_id": session_id,
        # token 自体は返さない方が安全。保存できたかのフラグだけ返す
        "token_saved": True
    })


# ====== /submit_captcha ======
@bp.route("/submit_captcha", methods=["POST"])
@login_required
def submit_captcha():
    from app.services.blog_signup.crypto_utils import encrypt
    from app.models import Site, ExternalBlogAccount
    from app.enums import BlogType
    from app.utils.captcha_dataset_utils import save_captcha_label_pair
    from app import db
    from flask import jsonify, session, request, current_app
    import logging, contextlib, asyncio
    import time

    logger = logging.getLogger(__name__)
    # ハンドオフ中は後片付けを抑止するフラグ
    keep_pw_session = False
    # finally で参照するので先に用意しておく（未定義参照対策）
    token = None
    captcha_text = request.form.get("captcha_text")
    if not captcha_text:
        return jsonify({"status": "error", "message": "CAPTCHA文字列が入力されていません"}), 400

    # 学習用保存
    img_name = session.get("captcha_image_filename")
    if captcha_text and img_name:
        with contextlib.suppress(Exception):
            save_captcha_label_pair(img_name, captcha_text)

    # ★ まずフォーム優先で受ける（ブラウザの Flask セッションが空でも復旧できる）
    site_id    = request.form.get("site_id", type=int) or session.get("captcha_site_id")
    account_id = request.form.get("account_id", type=int) or session.get("captcha_account_id")
    session_id = request.form.get("session_id") or session.get("captcha_session_id")

    # ★ サーバー側ストアから資格情報を復元（フォーム/Flaskセッションが空でもOK）
    cred = pw_get(session_id) if session_id else None

    email = (
        request.form.get("email")
        or session.get("captcha_email")
        or (cred and cred.get("email"))
    )
    password = (
        request.form.get("password")
        or session.get("captcha_password")
        or (cred and cred.get("password"))
    )
    livedoor_id = (
        request.form.get("livedoor_id")
        or session.get("captcha_nickname")
        or (cred and cred.get("livedoor_id"))
    )

    # URLサブドメイン=ユーザーID（希望値）。無ければ livedoor_id を使う
    desired_blog_id = (
        request.form.get("desired_blog_id")
        or request.form.get("blog_id")
        or request.form.get("sub")
        or session.get("captcha_desired_blog_id")
        or (cred and cred.get("desired_blog_id"))
        or livedoor_id
    )


    if not all([site_id, session_id, account_id]):
        return jsonify({"status": "error", "message": "セッション情報が不足しています"}), 400

    site = Site.query.get(site_id)
    acct = ExternalBlogAccount.query.get(account_id)
    if not site or not acct or acct.site_id != site_id:
        return jsonify({"status": "error", "message": "対象が不正です"}), 400
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        return jsonify({"status": "error", "message": "権限がありません"}), 403

    ok = False
    try:
        # ▶ 新API: 同一セッションで CAPTCHA 送信 → /register/done を待機
        ok = ld_submit_captcha(session_id, captcha_text)
    except Exception:
        logger.exception("[submit_captcha] CAPTCHA送信で例外")
        # セッションは後で必ず破棄
        return jsonify({"status": "error", "message": "CAPTCHA送信に失敗しました"}), 500

    if not ok:
        # 失敗時は中間アカウントをクリーンアップ（あれば）
        try:
            if acct and not getattr(acct, "atompub_key_enc", None):
                db.session.delete(acct)
                db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            # セッションキー掃除（captcha_status は残す）
            for k in list(session.keys()):
                if k.startswith("captcha_") and k != "captcha_status":
                    session.pop(k)
        return jsonify({
            "status": "recreate_required",
            "message": "CAPTCHA突破に失敗しました。もう一度お試しください。",
            "site_id": site_id,
        }), 200

    # --- ここから 既存の「メール認証→AtomPubキー回収」を継続 ---
    try:
        # メール確認リンク取得（最大 5 回 / 30 秒）
        # ★ 変数名の食い違いバグ修正：token を一元化して扱う
        token = (
            request.form.get("token")
            or session.get("captcha_token")
            or (cred and cred.get("token"))
        )
        if not token:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "recreate_required",
                "message": "確認メールのトークンが見つかりませんでした（セッション復元に失敗）",
                "site_id": site_id,
            }), 200
        activation_url = None  # ← これを追加
        for _ in range(5):
            with contextlib.suppress(Exception):
                activation_url = asyncio.run(poll_latest_link_gw(token))
            if activation_url:
                break
            time.sleep(6)

        if not activation_url:
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "recreate_required",
                "message": "確認メールリンクが取得できませんでした",
                "site_id": site_id,
            }), 200

        # Playwright セッションでそのまま認証URLへ遷移して final 入力を拾う
        # （recover_atompub_key はページを受け取って blog_id / api_key を抽出する実装）
        # reviveは基本不要だが、落ちていたら自動復帰
        page = pwctl.run(pwctl.get_page(session_id)) or pwctl.run(pwctl.revive(session_id))
        if not page:
            raise RuntimeError("Playwright セッションが消失しました")

        # 認証URLへ遷移（これも pwctl のループ上で）
        pwctl.run(page.goto(activation_url, wait_until="load"))

        # ★ ここを asyncio.run(...) ではなく pwctl.run(...) にするのがポイント
        # ★ 置換：recover で使う livedoor の user_id は、基本 livedoor_id を使う
        user_id = (
            request.form.get("livedoor_id")
            or request.form.get("user_id")
            or request.form.get("userid")
            or request.form.get("username")
            or request.form.get("account_id")
            or livedoor_id
        )
        if not user_id:
            current_app.logger.error("[submit_captcha] livedoor user_id is missing (sid=%s)", session_id)
            return jsonify({"ok": False, "error": "missing_user_id"}), 400

        
        # --- ここでフォーム値を集める（名称の揺れを吸収） ---
        nickname = (
            request.form.get("nickname")
            or request.form.get("display_name")
            or request.form.get("name")
        )

        # ここからは “既存値を優先し、未設定のときだけフォームから補完”
        email = (
            email
            or request.form.get("email")
            or request.form.get("livedoor_email")
            or request.form.get("mail")
        )

        password = (
            password
            or request.form.get("password")
            or request.form.get("livedoor_password")
            or request.form.get("pass")
        )

        # desired_blog_id は関数前半で cred/セッション/フォームから一度確定済み。
        # 後段で再計算・上書きしない（そのまま desired_blog_id を使う）。


        # 最低限のバリデーション（必要に応じて 400 を返す）
        if not email or not password:
            current_app.logger.error(
                "[submit_captcha] email/password missing (sid=%s, has_email=%s, has_pw=%s)",
                session_id, bool(email), bool(password)
            )
            return jsonify({"ok": False, "error": "missing_email_or_password"}), 400

        if not nickname:
            nickname = email.split("@")[0]  # フォールバック

        # ✅ 自動作成はやめて **手動ハンドオフ** に切替
        # 同一セッションの新タブで /member/blog/create を開くだけ（送信はしない）
        result = open_create_tab_for_handoff(
            session_id,
            site,
            prefill_title=True,   # 生成済みタイトルを入力欄にプリフィル
        )
        if not result or not result.get("ok"):
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            return jsonify({
                "status": "handoff_error",
                "message": result.get("error", "handoff_failed"),
                "site_id": site_id,
                "account_id": account_id
            }), 200

        # フロントはこのURLを新規タブで開く（ユーザーが「ブログを作成する」を手動クリック）
        handoff = {
            "url": result.get("url"),
            "prefilled_title": result.get("prefilled_title"),
            "has_blog_id_box": result.get("has_blog_id_box"),
            # ★ 追加：後続の /handoff_finalize で確実に同一PWセッションを掴めるようにする
            "session_id": session_id,
        }
        # ここからは人手作業にバトンを渡すので、セッションは維持する
        keep_pw_session = True
        current_app.logger.info(
            "[handoff] ready sid=%s url=%s has_id_box=%s title=%s",
            session_id,
            handoff.get("url"),
            handoff.get("has_blog_id_box"),
            handoff.get("prefilled_title"),
        )
        session["captcha_status"] = {
            "captcha_sent": True,
            "email_verified": True,
            "account_created": False,
            "api_key_received": False,
            "step": "handoff_ready",
            "site_id": site_id,
            "account_id": account_id,
            # ★ 追加：セッション側にも handoff.session_id を保存（保険）
            "handoff": handoff,
        }
        return jsonify({
            "status": "handoff_ready",
            "site_id": site_id,
            "account_id": account_id,
            "handoff": handoff,
            "next_cta": "open_create_ui"
        }), 200
        

    finally:
        # ハンドオフ中は何も片付けない（同一セッションで人手操作を続行するため）
        if keep_pw_session:
            current_app.logger.info("[cleanup] handoff in progress -> keep session alive (sid=%s)", session_id)
        else:
            current_app.logger.info("[cleanup] closing pw session & clearing temp keys (sid=%s)", session_id)
            with contextlib.suppress(Exception):
                pwctl.close_session(session_id)
            with contextlib.suppress(Exception):
                pw_clear(session_id)
            # メールトークンはハンドオフでない通常経路のみ解放
            # セマフォ解放は /external-seo/end に委ねる（ここではメールトークンは解放対象ではない）
            # もしここで解放したい場合は extseo_token を取り出して release する
            # with contextlib.suppress(Exception):
            #     ext_tok = session.get("extseo_token")
            #     if ext_tok:
            #         release(ext_tok)
            # 進捗オブジェクト(captcha_status)は残しつつ、一時キー(captcha_*)を掃除
            for key in list(session.keys()):
                if key.startswith("captcha_") and key != "captcha_status":
                    session.pop(key)

# ====== /handoff_finalize ======
@bp.route("/handoff_finalize", methods=["POST"])
@login_required
def handoff_finalize():
    """
    ユーザーが /member/blog/create で手動作成を終えたあとに呼ぶ。
    既存の Playwright セッションでダッシュボードから blog_id / api_key を回収して DB に保存する。
    """
    from flask import jsonify, session, request, current_app
    from app.models import Site, ExternalBlogAccount
    from app.enums import BlogType
    from app import db
    import contextlib
    # ★ 追加：ここでのみ使うためローカル import（明示しておく）
    from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key

    site_id = request.form.get("site_id", type=int) or (session.get("captcha_status") or {}).get("site_id")
    account_id = request.form.get("account_id", type=int) or (session.get("captcha_status") or {}).get("account_id")
    session_id = session.get("captcha_session_id") or (session.get("captcha_status") or {}).get("handoff", {}).get("session_id")
    # handoff_ready の時点では session["captcha_session_id"] を保持している前提
    if not session_id:
        # 念のため DB 側から拾う（あれば）
        acct = ExternalBlogAccount.query.get(account_id) if account_id else None
        if acct and getattr(acct, "captcha_session_id", None):
            session_id = acct.captcha_session_id

    if not all([site_id, account_id, session_id]):
        return jsonify({"status": "error", "message": "handoff セッション情報が不足しています"}), 400

    site = Site.query.get(site_id)
    acct = ExternalBlogAccount.query.get(account_id)
    if not site or not acct or acct.site_id != site_id:
        return jsonify({"status": "error", "message": "対象が不正です"}), 400
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        return jsonify({"status": "error", "message": "権限がありません"}), 403

    # ここから回収
    cred = pw_get(session_id) or {}
    email = cred.get("email") or session.get("captcha_email")
    password = cred.get("password") or session.get("captcha_password")
    livedoor_id = cred.get("livedoor_id") or session.get("captcha_nickname")
    desired_blog_id = cred.get("desired_blog_id") or session.get("captcha_desired_blog_id") or livedoor_id
    email_token = cred.get("token") or session.get("captcha_token")

    if not (email and password and livedoor_id):
        return jsonify({"status": "error", "message": "資格情報が不足しています"}), 400

    # 既存セッションからページを取得（落ちてたら revive）
    page = pwctl.run(pwctl.get_page(session_id)) or pwctl.run(pwctl.revive(session_id))
    if not page:
        return jsonify({"status": "error", "message": "Playwright セッションが見つかりません"}), 500

    # ここで blog_id / api_key を回収
    result = pwctl.run(recover_atompub_key(
        page,
        livedoor_id=livedoor_id,
        nickname=(email.split("@")[0] if email else livedoor_id),
        email=email,
        password=password,
        site=site,
        desired_blog_id=desired_blog_id,
    ))

    if not result or not result.get("success"):
        return jsonify({
            "status": "handoff_error",
            "message": result.get("error", "APIキーの回収に失敗しました"),
            "site_id": site_id,
            "account_id": account_id,
        }), 200

    new_blog_id  = (result.get("blog_id") or "").strip() or None
    new_api_key  = (result.get("api_key") or "").strip() or None
    new_endpoint = (result.get("endpoint") or "").strip() or None

    # 重複 blog_id があれば既存を優先
    dup = None
    if new_blog_id:
        dup = (ExternalBlogAccount.query
               .filter(
                   ExternalBlogAccount.site_id == site_id,
                   ExternalBlogAccount.blog_type == (acct.blog_type or BlogType.LIVEDOOR),
                   ExternalBlogAccount.livedoor_blog_id == new_blog_id,
                   ExternalBlogAccount.id != account_id
               )
               .first())
    target = dup or acct

    if hasattr(target, "is_captcha_completed"):
        target.is_captcha_completed = True
    if new_blog_id and hasattr(target, "livedoor_blog_id"):
        target.livedoor_blog_id = new_blog_id
    if new_blog_id and hasattr(target, "username"):
        if not target.username or target.username.startswith("u-"):
            target.username = new_blog_id
    if new_endpoint and hasattr(target, "atompub_endpoint"):
        with contextlib.suppress(Exception):
            target.atompub_endpoint = new_endpoint
    if new_api_key and hasattr(target, "atompub_key_enc"):
        from app.services.blog_signup.crypto_utils import encrypt
        with contextlib.suppress(Exception):
            target.atompub_key_enc = encrypt(new_api_key)
        if hasattr(target, "api_post_enabled"):
            target.api_post_enabled = True

    db.session.commit()

    got_api = bool(new_api_key or getattr(target, "atompub_key_enc", None))
    resolved_account_id = target.id
    session["captcha_status"] = {
        "captcha_sent": True,
        "email_verified": True,
        "account_created": True,
        "api_key_received": got_api,
        "step": "API取得完了",
        "site_id": site_id,
        "account_id": resolved_account_id,
    }

    # handoff 完了したので後片付け
    with contextlib.suppress(Exception):
        pwctl.close_session(session_id)
    with contextlib.suppress(Exception):
        pw_clear(session_id)
    # 同上：セマフォは /external-seo/end で解放。ここでは何もしない
    # with contextlib.suppress(Exception):
    #     ext_tok = session.get("extseo_token")
    #     if ext_tok:
    #         release(ext_tok)
    for key in list(session.keys()):
        if key.startswith("captcha_") and key != "captcha_status":
            session.pop(key)

    return jsonify({
        "status": "captcha_success",
        "step": session["captcha_status"]["step"],
        "site_id": site_id,
        "account_id": resolved_account_id,
        "api_key_received": got_api,
        "next_cta": "ready_to_post" if got_api else "captcha_done"
    }), 200


@bp.route("/ld/open_create_ui", methods=["POST"])
@login_required
def open_create_ui():
    """任意タイミングで“別タブで作成画面”を開きたい場合の軽量API（UIボタン用）"""
    from flask import request, jsonify, session as flask_session
    from app.models import Site
    session_id = request.form.get("session_id") or flask_session.get("captcha_session_id")
    site_id    = request.form.get("site_id")    or flask_session.get("captcha_site_id")
    if not session_id or not site_id:
        return jsonify({"ok": False, "error": "missing_params"}), 400
    site = Site.query.get(int(site_id))
    if not site:
        return jsonify({"ok": False, "error": "site_not_found"}), 404
    result = open_create_tab_for_handoff(session_id, site, prefill_title=True)
    if not result or not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error", "handoff_failed")}), 500
    return jsonify({"ok": True, **result}), 200

@bp.route("/captcha_status", methods=["GET"])
@login_required
def get_captcha_status():
    from flask import session, jsonify, request
    # DBフォールバック用
    from app.models import ExternalBlogAccount

    status = session.get("captcha_status")

    # 任意：?account_id=... が来たら整合性チェック
    q_acc = request.args.get("account_id", type=int)

    # セッションがある場合の基本応答
    if status:
        if q_acc and status.get("account_id") and status["account_id"] != q_acc:
            # 別アカウントのステータスを見に来た場合は未開始扱い
            return jsonify({"status": "not_started", "step": "未開始"}), 200
        return jsonify(status), 200

    # ★ セッションが切れても、DBがAPI取得済なら「API取得完了」を返すフォールバック
    if q_acc:
        acct = ExternalBlogAccount.query.get(q_acc)
        if acct and getattr(acct, "atompub_key_enc", None):
            return jsonify({
                "captcha_sent": True,
                "email_verified": True,          # ここは推定（API取得済み前提）
                "account_created": True,         # 同上
                "api_key_received": True,
                "step": "API取得完了",
                "site_id": getattr(acct, "site_id", None),
                "account_id": q_acc
            }), 200

    # 何も情報がない
    return jsonify({"status": "not_started", "step": "未開始"}), 200

@bp.get("/generate")
@login_required
def external_seo_generate_get():
    from datetime import datetime, timezone
    from app import db
    from app.models import Site, ExternalBlogAccount, BlogType
    from app.tasks import enqueue_generate_and_schedule
    from sqlalchemy import and_

    site_id = request.args.get("site_id", type=int)
    account_id = request.args.get("account_id", type=int)
    blog_type_param = request.args.get("blog_type", default="livedoor").strip().lower() if request.args.get("blog_type") else "livedoor"

    if not site_id:
        flash("site_id が不足しています。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    try:
        target_blog_type = getattr(BlogType, blog_type_param.upper())
    except Exception:
        target_blog_type = BlogType.LIVEDOOR

    # 対象アカウントの選定
    if account_id:
        acct = ExternalBlogAccount.query.get_or_404(account_id)
        if acct.site_id != site_id:
            flash("不正なアクセスです（サイト不一致）", "danger")
            return redirect(url_for("main.external_seo_sites"))
        if acct.blog_type != target_blog_type:
            flash("不正なアクセスです（プラットフォーム不一致）", "danger")
            return redirect(url_for("main.external_seo_sites"))
        if not acct.atompub_key_enc:
            flash("このアカウントはAPIキー未取得のため記事生成できません。", "danger")
            return redirect(url_for("main.external_seo_sites"))
        accounts_to_run = [acct]
    else:
        # まとめ実行：未ロック & API 取得済みのみ候補にする
        accounts_to_run = (
            ExternalBlogAccount.query
            .filter(
                and_(
                    ExternalBlogAccount.site_id == site_id,
                    ExternalBlogAccount.blog_type == target_blog_type,
                    ExternalBlogAccount.atompub_key_enc.isnot(None),
                    ExternalBlogAccount.generation_locked.is_(False),
                )
            )
            .all()
        )
        if not accounts_to_run:
            flash("実行可能なアカウントが見つかりません（API未取得 または 既にロック済み）。", "warning")
            return redirect(url_for("main.external_seo_sites"))

    ok, ng, skipped_locked = 0, 0, 0
    failed = []

    for acct in accounts_to_run:
        try:
            # ---- ここが恒久ロックの肝 ----
            # 行ロックを取り、二重実行を防ぐ
            row = (
                ExternalBlogAccount.query
                .with_for_update()           # SELECT ... FOR UPDATE
                .filter_by(id=acct.id)
                .first()
            )
            if not row:
                skipped_locked += 1
                continue

            # 既にロック済みならスキップ
            if row.generation_locked:
                skipped_locked += 1
                continue

            # ここで恒久ロックを立てて確定
            row.generation_locked = True
            row.generation_locked_at = datetime.now(timezone.utc)
            db.session.add(row)
            db.session.commit()             # 先に確定 → 以後は二重実行不可

            # ロック確定後にキュー投入
            enqueue_generate_and_schedule(
                user_id=current_user.id,
                site_id=site_id,
                blog_account_id=row.id,
                count=100,
                per_day=10,
                start_day_jst=None,   # 翌日開始（関数内のデフォルトで処理）
            )
            ok += 1

        except Exception as e:
            db.session.rollback()
            ng += 1
            failed.append((acct.id, str(e)))

    # フィードバック
    if ok and not ng:
        msg = f"{ok}件のブログで記事生成を開始"
        if skipped_locked:
            msg += f" ／ ロック済みスキップ {skipped_locked}件"
        flash(msg, "success")
    elif ok and ng:
        flash(f"{ok}件開始 / {ng}件失敗（ロック済みスキップ {skipped_locked}件）", "warning")
    else:
        # 1件も開始できなかった
        if skipped_locked:
            flash("すべての対象がロック済みのため実行されませんでした。", "warning")
        else:
            flash("記事生成の開始に失敗しました。", "danger")

    if failed:
        for aid, msg in failed[:3]:
            flash(f"account_id={aid}: {msg}", "danger")
        if len(failed) > 3:
            flash(f"…他 {len(failed)-3}件", "danger")

    return redirect(url_for("main.external_seo_sites"))


from flask import render_template, redirect, url_for, request, session, flash
from app.services.mail_utils.mail_tm import poll_latest_link_tm_async as poll_latest_link_gw
from app.services.blog_signup.livedoor_signup import extract_verification_url

@bp.route('/confirm_email_manual/<task_id>')
def confirm_email_manual(task_id):
    """
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面。
    """
    # メール受信（最大30回ポーリング） ← 既存関数を再利用
    email_body = poll_latest_link_gw(task_id=task_id, max_attempts=30, interval=5)

    if email_body:
        # 認証URLを抽出
        verification_url = extract_verification_url(email_body)
        if verification_url:
            return render_template("confirm_email.html", verification_url=verification_url)
        else:
            flash("認証リンクが見つかりませんでした", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("認証メールを取得できませんでした", "danger")
        return redirect(url_for('dashboard'))

from flask import request, session, redirect, url_for, flash
from app.services.blog_signup.livedoor_signup import fetch_livedoor_credentials


@bp.route('/finish_signup/<task_id>', methods=['POST'])
def finish_signup(task_id):
    """
    メール認証が完了した後に呼ばれる処理。
    AtomPub API Keyを取得し、DB保存 or 表示に進む。
    """
    try:
        # すでに存在する task_id のセッションや保存情報から再開
        result = fetch_livedoor_credentials(task_id)

        if result and result.get("blog_id") and result.get("api_key"):
            # 必要に応じてDB保存 or セッションに保存（ここでは表示用）
            flash("🎉 AtomPub API情報を正常に取得しました", "success")
            flash(f"ブログID: {result['blog_id']}", "info")
            flash(f"API Key: {result['api_key']}", "info")
            return redirect(url_for('dashboard'))  # または account_details, etc.
        else:
            flash("API情報の取得に失敗しました", "danger")
            return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f"エラーが発生しました: {str(e)}", "danger")
        return redirect(url_for('dashboard'))

from flask import render_template, abort
from app.services.blog_signup.livedoor_signup import fetch_livedoor_credentials

@bp.route("/external/livedoor/confirm/<task_id>")
def confirm_livedoor_email(task_id):
    creds = fetch_livedoor_credentials(task_id)
    if not creds:
        abort(404, description="認証情報が見つかりません")
    return render_template("confirm_email.html", blog_id=creds["blog_id"], api_key=creds["api_key"])

# ===============================
# 外部SEO記事生成ルート（新規追加）
# ===============================

from flask import Blueprint, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import ExternalBlogAccount
#from app.tasks import _run_external_post_job

# 既存の
# @bp.route("/external-seo/generate/<int:site_id>/<int:blog_id>", methods=["POST"])
# def external_seo_generate(...):
# を丸ごと置き換え

@bp.route("/external-seo/generate/<int:site_id>/<int:blog_id>", methods=["POST"])
@login_required
def external_seo_generate(site_id, blog_id):
    """
    既存の /external-seo/generate/<site_id>/<blog_id> を温存したまま、
    生成＆スケジューリングの新ロジックに差し替え。
    - 100本生成
    - 1日10本
    - スケジュール開始は「生成開始の翌日」
    """
    from flask import redirect, url_for, flash
    from app.models import ExternalBlogAccount, Site, BlogType
    from app.external_seo_generator import generate_and_schedule_external_articles

    # アカウント取得
    acct = ExternalBlogAccount.query.get_or_404(blog_id)

    # site_id整合性
    if acct.site_id != site_id:
        flash("不正なアクセスです（サイト不一致）。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    # 所有権チェック（管理者はスキップ）
    site = Site.query.get_or_404(site_id)
    if (not current_user.is_admin) and (site.user_id != current_user.id):
        abort(403)

    # APIキー必須
    if not getattr(acct, "atompub_key_enc", None):
        flash("APIキーが未取得のため記事生成できません。", "danger")
        return redirect(url_for("main.external_seo_sites"))

    try:
        # ※ start_day_jst を省略 → ジェネレータ側で「翌日開始」に自動化
        created = generate_and_schedule_external_articles(
            user_id=current_user.id,
            site_id=site_id,
            blog_account_id=acct.id,
            count=100,
            per_day=10,
            start_day_jst=None,
        )
        flash(f"外部SEO記事の生成を開始しました（{created}件、1日10本・翌日から投稿）。", "success")
    except Exception as e:
        current_app.logger.exception("[external-seo] generate (legacy route) failed")
        flash(f"記事生成開始に失敗しました: {e}", "danger")

    return redirect(url_for("main.external_seo_sites"))


# ===============================
# 外部SEO: 100本生成＋1日10本スケジューリング（新規）
# ===============================
from flask import request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime, timedelta, timezone
from app.models import ExternalBlogAccount, BlogType, Article
from app.external_seo_generator import generate_and_schedule_external_articles
from sqlalchemy import or_


JST = timezone(timedelta(hours=9))

@bp.route("/external-seo/generate_and_schedule", methods=["POST"])
@login_required
def external_seo_generate_and_schedule():
    """
    外部SEO記事をまとめて生成し、1日10本（JST 10:00〜21:59の“切りの良くない分”）でスケジューリング。
    JSON/FORM:
      site_id: int (必須)
      blog_account_id: int (任意。未指定なら site_id に紐づく最新 Livedoor を自動選択)
      count: 生成本数（デフォルト100）
      per_day: 1日あたり本数（デフォルト10）
      start_date_jst: "YYYY-MM-DD"（JSTの開始日。省略時は当日）
    """
    # 入力パラメータ
    site_id = request.form.get("site_id", type=int) or (request.json or {}).get("site_id")
    count = request.form.get("count", type=int) or (request.json or {}).get("count", 100)
    per_day = request.form.get("per_day", type=int) or (request.json or {}).get("per_day", 10)
    start_date_s = request.form.get("start_date_jst") or (request.json or {}).get("start_date_jst")

    if not site_id:
        return jsonify({"ok": False, "error": "site_id is required"}), 400
    
    # ▼ 通常記事（外部SEO以外で投稿済み）が 50 本未満なら実行をブロック
    normal_count = (
        Article.query
        .filter(Article.site_id == site_id)
        .filter(or_(Article.source.is_(None), Article.source != "external"))
        .filter(Article.status.in_(["posted", "published"]))  # ← done を含めない
        .count()
    )
    if normal_count < 100:
        return jsonify({
            "ok": False,
            "error": "外部SEO開始の条件を満たしてません",
            "count": normal_count
        }), 400

    if start_date_s:
        try:
            y, m, d = map(int, start_date_s.split("-"))
            start_day_jst = datetime(y, m, d, tzinfo=JST)
        except Exception:
            return jsonify({"ok": False, "error": "start_date_jst must be YYYY-MM-DD"}), 400
    else:
        start_day_jst = datetime.now(JST).replace(hour=0, minute=0, second=0, microsecond=0)

    # 対象アカウント
    blog_account_id = request.form.get("blog_account_id") or (request.json or {}).get("blog_account_id")
    if blog_account_id:
        acct = ExternalBlogAccount.query.get(int(blog_account_id))
    else:
        acct = (ExternalBlogAccount.query
                .filter_by(site_id=site_id, blog_type=BlogType.LIVEDOOR)
                .order_by(ExternalBlogAccount.id.desc())
                .first())
    if not acct:
        return jsonify({"ok": False, "error": "Livedoorアカウントが見つかりません"}), 400

    # 所有権チェック（管理者はスキップ）
    if (not current_user.is_admin) and (acct.site.user_id != current_user.id):
        return jsonify({"ok": False, "error": "権限がありません"}), 403

    # 実行
    try:
        created = generate_and_schedule_external_articles(
            user_id=current_user.id,
            site_id=site_id,
            blog_account_id=acct.id,
            count=int(count),
            per_day=int(per_day),
            start_day_jst=start_day_jst,
        )
        return jsonify({"ok": True, "created": created})
    except Exception as e:
        current_app.logger.exception("[external-seo] generate_and_schedule failed")
        return jsonify({"ok": False, "error": str(e)}), 500

from sqlalchemy.exc import IntegrityError
import secrets, time  # ★ 追加
import re as _re
from urllib.parse import urlparse
try:
    from unidecode import unidecode  # あれば日本語→ローマ字化
except Exception:
    def unidecode(x): return x


@bp.route("/external-seo/new-account", methods=["POST"])
@bp.route("/external-seo/new-account/", methods=["POST"])
@login_required
def external_seo_new_account():
    """
    Livedoorの仮アカウントを1件作成（必須カラムは存在確認してからセット）。
    例外時も必ずJSONで返す。
    """
    from flask import request, jsonify
    from app.models import Site, ExternalBlogAccount, BlogType, Article
    from app import db
    from sqlalchemy import or_
    import logging
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # ---- ユーティリティ -------------------------------------------------
    def _stub_email(site_id: int) -> str:
        """email UNIQUE対策：衝突しないダミーを毎回生成"""
        # 例: pending-12-1723358300123-a3f1@stub.local
        return f"pending-{site_id}-{int(time.time()*1000)}-{secrets.token_hex(2)}@stub.local"

    def _stub_name(prefix: str, site_id: int) -> str:
        """username 用のダミー（安全にユニーク寄りに）"""
        # 例: u-12-1723358300123-a
        return f"{prefix}-{site_id}-{int(time.time()*1000)}-{secrets.token_hex(1)}"

    def _slugify_from_site(site: "Site") -> str:
        """
        サイト名/URLから display 用の短いスラッグを生成（a-z0-9-、先頭は英字、最大20文字）
        外部アカウントのカード表示に使う。DBの一意制約には関与しない。
        """
        base = (site.name or "")[:60]
        if not base and getattr(site, "url", None):
            try:
                host = urlparse(site.url).hostname or ""
                base = host.split(".")[0] if host else ""
            except Exception:
                base = ""

        if not base:
            base = f"site-{site.id}"

        s = unidecode(str(base)).lower()
        s = s.replace("&", " and ")
        s = _re.sub(r"[^a-z0-9]+", "-", s)
        s = _re.sub(r"-{2,}", "-", s).strip("-")
        if not s:
            s = f"site-{site.id}"
        if s[0].isdigit():
            s = "blog-" + s
        s = s[:20]
        if len(s) < 3:
            s = (s + "-blog")[:20]
        return s
    # --------------------------------------------------------------------

    try:
        site_id = request.form.get("site_id", type=int)
        if not site_id:
            return jsonify({"ok": False, "error": "site_id がありません"}), 200

        site = Site.query.get(site_id)
        if not site:
            return jsonify({"ok": False, "error": "Site が見つかりません"}), 200
        if (not current_user.is_admin) and (site.user_id != current_user.id):
            return jsonify({"ok": False, "error": "権限がありません"}), 200
        
        # ▼ 通常記事（外部SEO以外で WPに投稿済み）が 100 本未満ならブロック
        normal_count = (
            Article.query
            .filter(Article.site_id == site_id)
            .filter(or_(Article.source.is_(None), Article.source != "external"))
            .filter(Article.status.in_(["posted", "published"]))  # ← done を含めない
            .count()
        )
        if normal_count < 100:
            return jsonify({
                "ok": False,
                "error": "外部SEO開始の条件を満たしてません",
                "count": normal_count
            }), 400

        # 表示用スラッグ（カードのタイトルに使う）
        display_slug = _slugify_from_site(site)

        # UNIQUE衝突に備えて数回だけリトライ
        attempts = 0
        while True:
            try:
                # まず最小限の必須だけでインスタンス化（存在しない列は触らない）
                acc = ExternalBlogAccount(
                    site_id=site.id,
                    blog_type=BlogType.LIVEDOOR,
                )

                # --- カラムが存在する場合のみ安全にセット ---
                # UNIQUE の可能性がある email は必ずユニークなダミーにする
                if hasattr(acc, "email"):
                    acc.email = _stub_email(site.id)

                # username はダミー、nickname は表示に近い値（サイト由来スラッグ）を入れておく
                if hasattr(acc, "username"):
                    acc.username = _stub_name("u", site.id)
                if hasattr(acc, "password"):
                    acc.password = ""  # 仮
                if hasattr(acc, "nickname"):
                    acc.nickname = display_slug  # ← ここをサイト由来に

                # 状態系（存在すれば）
                if hasattr(acc, "status"):                acc.status = "active"
                if hasattr(acc, "message"):               acc.message = None
                if hasattr(acc, "cookie_path"):           acc.cookie_path = None
                if hasattr(acc, "livedoor_blog_id"):      acc.livedoor_blog_id = None
                if hasattr(acc, "atompub_key_enc"):       acc.atompub_key_enc = None
                if hasattr(acc, "api_post_enabled"):      acc.api_post_enabled = False
                if hasattr(acc, "is_captcha_completed"):  acc.is_captcha_completed = False
                # is_email_verified は存在しない環境があるため触らない
                if hasattr(acc, "posted_cnt"):            acc.posted_cnt = 0
                if hasattr(acc, "next_batch_started"):    acc.next_batch_started = None
                if hasattr(acc, "created_at"):            acc.created_at = datetime.utcnow()

                db.session.add(acc)
                db.session.commit()
                break  # ← 成功

            except IntegrityError:
                # email（や他の一意制約）衝突時は再採番してリトライ
                db.session.rollback()
                attempts += 1
                if attempts >= 5:
                    logger.exception("[external_seo_new_account] integrity error (retries exceeded)")
                    return jsonify({"ok": False, "error": "DBの一意制約で作成に失敗しました。時間をおいて再試行してください。"}), 200
                # ループ先頭で新しいダミーを採番して再作成

        account_payload = {
            "id": acc.id,
            # 表示名はサイト名ベースのスラッグを使用（カードのタイトルが人間にわかりやすくなる）
            "blog_title": display_slug,
            "public_url": None,
            "api_key": None,
            "stat_total": 0,
            "stat_posted": 0,
        }
        return jsonify({"ok": True, "site_id": site.id, "account": account_payload}), 200

    except Exception as e:
        db.session.rollback()
        logger.exception("[external_seo_new_account] error")
        return jsonify({"ok": False, "error": f"サーバエラー: {str(e)}"}), 200
    

# ────────── 外部SEOステータスAPI（統合版：これ1本だけ残す） ──────────
from app.utils.semaphore import current_active, try_acquire, release, LIMIT

@bp.get("/external-seo/status")
@login_required
def external_seo_status():
    """
    フロントのポーリング用APIを一本化：
      - 並列実行の使用状況（active/limit/available）
      - extseo_token に紐づく進捗・captcha_url 等の状態
    を同時に返す。フロントは必要なキーだけ見ればOK（下位互換）。
    """
    from flask import jsonify, session

    # 1) 並列実行の容量情報
    active = current_active()
    cap = {
        "active": active,
        "limit": LIMIT,
        "available": max(LIMIT - active, 0),
    }

    # 2) トークンに紐づく進捗（あれば返す）
    tok = session.get("extseo_token")
    st = {}
    # EXTSEO_STATUS はファイル先頭などで dict 初期化済み想定
    try:
        st = EXTSEO_STATUS.get(tok) or {}
    except Exception:
        st = {}

    # セッションにだけ積んでいるUI用の軽い進捗があればマージ（任意）
    try:
        sess_st = session.get("captcha_status") or {}
        if sess_st:
            st = {**st, **sess_st}
    except Exception:
        pass

    # まとめて返す（下位互換：従来のキーもそのまま st に含める）
    resp = {"ok": True, **cap, **st}
    if not tok:
        resp["token_missing"] = True
    return jsonify(resp)



@bp.route("/external-seo/start", methods=["POST"])
@login_required
def external_seo_start():
    token = try_acquire()
    if not token:
        flash("外部SEOの同時実行は最大3件までです。しばらく待ってから再度お試しください。", "error")
        return jsonify({
            "ok": False,
            "reason": "busy",
            "message": "外部SEO実行が混雑中です"
        }), 429

    # ★ セマフォ用トークンをセッションに保存（captcha_tokenとは分離）
    session["extseo_token"] = token
    return jsonify({"ok": True, "token": token})

# ==== 追加: 外部SEO ブートストラップ（サーバが従来規則で値を生成し、ヘルパーに渡す）====
@bp.post("/external-seo/bootstrap")
@login_required
def external_seo_bootstrap():
    from flask import request, jsonify, session
    from flask_login import current_user
    from app import db
    from app.models import Site
    from app.services.mail_utils.mail_gw import create_inbox
    # 生成規則は従来のまま流用
    from app.services.blog_signup.livedoor_signup import (
        generate_safe_id, generate_safe_password, suggest_livedoor_blog_id,
        _craft_blog_title as ld_craft_blog_title  # 私有関数だが import 可。規則完全一致のため使用
    )

    # /external-seo/start で配られた extseo_token を必須とする
    tok = session.get("extseo_token")
    if not tok:
        return jsonify({"ok": False, "error": "missing extseo_token; call /external-seo/start first"}), 400

    # 入力（JSON or form 両対応）
    if request.is_json:
        site_id = request.json.get("site_id")
        account_id = request.json.get("account_id")
    else:
        site_id = request.form.get("site_id")
        account_id = request.form.get("account_id")

    try:
        site_id = int(site_id) if site_id is not None else None
    except Exception:
        site_id = None
    try:
        account_id = int(account_id) if account_id is not None else None
    except Exception:
        account_id = None

    if not site_id:
        return jsonify({"ok": False, "error": "missing site_id"}), 400

    site = Site.query.get(site_id)
    if not site or (not current_user.is_admin and site.user_id != current_user.id):
        return jsonify({"ok": False, "error": "permission denied"}), 403

    # ▼ 従来と同じ規則で生成（＝VPS時代と完全一致）
    email, inbox_token = create_inbox()                 # 既存GWのまま
    # 英字開始・3–20・英数＋_ 準拠の候補ロジックを採用
    livedoor_id = generate_livedoor_id_candidates(site)[0]
    password    = generate_safe_password()
    try:
        blog_title = ld_craft_blog_title(site)          # タイトル規則を完全踏襲
    except Exception:
        blog_title = "ブログ"
    try:
        desired_blog_id = suggest_livedoor_blog_id(site.name or site.url or "", db.session)
    except Exception:
        desired_blog_id = None

    # UI ポーリング用の軽い初期状態
    st = dict(session.get("captcha_status") or {})
    st.update({
        "step": "bootstrap_ok",
        "progress": max(5, int(st.get("progress") or 0)),
        "site_id": site_id,
        "account_id": account_id,
    })
    session["captcha_status"] = st

    # 絶対URL（Blueprint 名に依存せず、確実に解決）
    root = request.url_root.rstrip("/")
    callback_url = f"{root}/external-seo/callback"
    upload_url   = f"{root}/external-seo/prepare_captcha"
    # STEP 3 で実装予定。先に URL を返しておき、ヘルパーは存在すれば使う
    verify_poll_url = f"{root}/external-seo/fetch_verify_url?token={inbox_token}"

    return jsonify({
        "ok": True,
        "token": tok,
        "site_id": site_id,
        "account_id": account_id,
        # 従来規則で生成した値（＝ヘルパーは“受け取ったまま”使う）
        "email": email,
        "inbox_token": inbox_token,
        "livedoor_id": livedoor_id,
        "password": password,
        "blog_title": blog_title,
        "desired_blog_id": desired_blog_id,
        # ヘルパーが叩くサーバ側の入口
        "callback_url": callback_url,
        "upload_url": upload_url,
        # メール認証URLの取得（STEP 3でサーバ実装。無ければヘルパーは自前fallback）
        "verify_poll_url": verify_poll_url,
    })

# ==== 追加: 外部SEO ステータスポーリング ====
@bp.get("/external-seo/captcha_status")
@login_required
def external_seo_captcha_status():
    """
    UIが定期ポーリングして進捗やCAPTCHA画像URL、完了フラグを取得する。
    セッションが見えない環境では空に近いレスポンスになるが、それでOK。
    """
    from flask import session, jsonify

    st = dict(session.get("captcha_status") or {})

    # 既定値（UI側での扱いを安定させる）
    st.setdefault("step", "idle")
    try:
        st["progress"] = max(0, min(100, int(st.get("progress") or 0)))
    except Exception:
        st["progress"] = 0

    # よく使うフィールドは必ず鍵を用意しておく（undefined回避）
    st.setdefault("captcha_url", None)
    st.setdefault("captcha_sent", False)
    st.setdefault("email_verified", False)
    st.setdefault("account_created", False)
    st.setdefault("api_key_received", False)
    st.setdefault("site_id", None)
    st.setdefault("account_id", None)

    # extseo_token が生きているか（並列ガードの可視化）
    st["extseo_active"] = bool(session.get("extseo_token"))

    return jsonify({"ok": True, **st})


# ==== 追加: 外部SEO メール認証URLのポーリング ====
@bp.get("/external-seo/fetch_verify_url")
def external_seo_fetch_verify_url():
    """
    ヘルパー（ユーザーPC）からポーリングされる。
    サーバ側（VPS時代と同じメール受信ロジック）で最新の認証メール本文を取得し、
    Livedoorの verify リンクを抽出して返す。見つからなければ ok:false。
    クライアントは一定間隔で再ポーリングする想定。
    例: GET /external-seo/fetch_verify_url?token=<inbox_token>&timeout=120&interval=5
    """
    from flask import request, jsonify, current_app
    # 既存：livedoor_signup から従来の抽出ロジックをそのまま使用（完全踏襲）
    from app.services.blog_signup.livedoor_signup import (
        extract_verification_url,          # 本文から verify URL 抜き出し
        poll_latest_link_gw,               # = mail_tm の poll を再輸出（VPS時代の実装）
    )

    token = (request.args.get("token") or "").strip()
    if not token:
        return jsonify({"ok": False, "error": "missing token"}), 400

    # デフォルトは 120秒/5秒間隔（VPS時代の体感に合わせる）
    try:
        timeout_sec = int(request.args.get("timeout", 120))
    except Exception:
        timeout_sec = 120
    try:
        interval_sec = int(request.args.get("interval", 5))
    except Exception:
        interval_sec = 5

    # poll_latest_link_gw は “メール本文テキスト” を返す想定（従来互換）
    # task_id=token をキーに、timeout/interval に応じてリトライ
    try:
        email_body = poll_latest_link_gw(
            task_id=token,
            max_attempts=max(1, int(timeout_sec // max(1, interval_sec))),
            interval=max(1, interval_sec),
        )
    except Exception as e:
        current_app.logger.exception("[EXTSEO-VERIFY] poll error: %s", e)
        return jsonify({"ok": False, "error": "poll_error"}), 500

    if not email_body:
        # まだ届かないだけ。ポーリング継続させる
        return jsonify({"ok": False, "reason": "no_mail"})

    # 本文から Livedoor の verify URL を抽出（規則は livedoor_signup.extract_verification_url に完全準拠）
    url = extract_verification_url(email_body)
    if not url:
        return jsonify({"ok": False, "reason": "no_link"})

    # 見つかった → 返す（ヘルパーが“ユーザーIPで”このURLにアクセスして認証を完了させる）
    return jsonify({"ok": True, "verification_url": url})


@bp.route("/external-seo/end", methods=["POST"])
@login_required
def external_seo_end():
    # ★ JSON bodyからではなくセッションに保存したextseo_tokenを解放する
    token = session.pop("extseo_token", None)
    if not token:
        return jsonify({"ok": False, "error": "no active external-seo token"}), 400
    release(token)
    return jsonify({"ok": True})

# -----------------------------------------------------------------
# 外部SEO: ローカルヘルパー → サーバー 進捗/完了コールバック
# -----------------------------------------------------------------
@bp.post("/external-seo/callback")
def external_seo_callback():
    """
    ローカルヘルパーからの進捗/完了通知。
    """
    from flask import request, jsonify, session, current_app
    from app import db
    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import encrypt

    data = request.get_json(silent=True) or {}
    tok  = (data.get("token") or "").strip()
    if not tok:
        return jsonify({"ok": False, "error": "missing token"}), 400

    # 可能ならブラウザセッションのトークンと突き合わせ（ズレても致命ではない）
    try:
        if session.get("extseo_token") and session["extseo_token"] != tok:
            current_app.logger.warning("[EXTSEO-CB] token mismatch (session present but different)")
    except Exception:
        pass

    # 型を整える（str "46" と int 46 の不一致で誤警告が出ないように）
    def _to_int(v):
        try:
            return int(v)
        except Exception:
            return v

    site_id    = _to_int(data.get("site_id"))
    account_id = _to_int(data.get("account_id"))

    step      = (data.get("step") or data.get("status") or "").strip()
    progress  = data.get("progress")
    helper_host = data.get("helper_host")
    helper_ip_public = data.get("helper_ip_public")
    blog_id   = (data.get("blog_id") or "").strip() or None
    endpoint  = (data.get("endpoint") or "").strip() or None
    api_key   = (data.get("api_key") or "").strip() or None

    # 進捗ログ
    try:
        current_app.logger.info(
            "[EXTSEO-CB] tok ok, site=%s acc=%s step=%s prog=%s helper_host=%s helper_ip=%s",
            site_id, account_id, step, progress, helper_host, helper_ip_public
        )
    except Exception:
        pass

    # ✅ まずトークンストアを更新（UIは /external-seo/status で読む）
    _extseo_update(tok,
                   step=step or None,
                   progress=progress if isinstance(progress, (int, float)) else None,
                   site_id=site_id,
                   account_id=account_id,
                   blog_id=blog_id,
                   endpoint=endpoint,
                   api_key_received=True if api_key else None)

    # account_id が無ければ進捗だけ受け付けて終了
    if not account_id:
        return jsonify({"ok": True, "noted": True})

    # --- DB 反映（APIキー・blog_idなどが来た場合） ---
    acct = ExternalBlogAccount.query.get(account_id)
    if not acct:
        return jsonify({"ok": False, "error": "account not found"}), 404

    # site_id の整合チェック（型合わせ済み）
    try:
        if site_id is not None and getattr(acct, "site_id", None) is not None and int(acct.site_id) != int(site_id):
            current_app.logger.warning("[EXTSEO-CB] site/account mismatch: site_id=%s acc.site_id=%s",
                                       site_id, acct.site_id)
    except Exception:
        pass

    touched = False

    if blog_id and hasattr(acct, "livedoor_blog_id"):
        try:
            acct.livedoor_blog_id = blog_id
            if hasattr(acct, "username") and (not acct.username or str(acct.username).startswith("u-")):
                acct.username = blog_id
            touched = True
        except Exception:
            db.session.rollback()
            return jsonify({"ok": False, "error": "failed to save blog_id"}), 500

    if endpoint and hasattr(acct, "atompub_endpoint"):
        try:
            acct.atompub_endpoint = endpoint
            touched = True
        except Exception:
            db.session.rollback()
            return jsonify({"ok": False, "error": "failed to save endpoint"}), 500

    if api_key and hasattr(acct, "atompub_key_enc"):
        try:
            acct.atompub_key_enc = encrypt(api_key)
            if hasattr(acct, "api_post_enabled"):
                acct.api_post_enabled = True
            if hasattr(acct, "is_captcha_completed"):
                acct.is_captcha_completed = True
            touched = True
        except Exception:
            db.session.rollback()
            return jsonify({"ok": False, "error": "failed to save api_key"}), 500

    if touched:
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            return jsonify({"ok": False, "error": "db commit failed"}), 500

    # セマフォ解放判定（省略可。既存実装があればそのまま）
    try:
        step_l = (step or "").lower()
        prog_i = None
        if isinstance(progress, (int, float)):
            try:
                prog_i = int(progress)
            except Exception:
                prog_i = None
        should_release = (
            step_l in {"apikey_received", "api_key_ok", "done", "complete", "failed", "error"}
            or bool(api_key)
            or (prog_i is not None and prog_i >= 100)
        )
        if should_release and tok:
            try:
                release(tok)  # 既存の try_acquire に対応
                current_app.logger.info("[EXTSEO-CB] released semaphore token")
            except Exception as e:
                current_app.logger.exception("[EXTSEO-CB] release token failed: %s", e)
            if session.get("extseo_token") == tok:
                session.pop("extseo_token", None)
    except Exception:
        pass

    return jsonify({"ok": True})



# --- 外部SEO: クライアントヘルパーがCAPTCHA画像をアップロードする受け口 ---
@bp.post("/external-seo/prepare_captcha")
def external_seo_prepare_captcha_upload():
    """
    クライアント（127.0.0.1のヘルパー）が撮ったCAPTCHA画像をアップロード。
    期待: multipart/form-data で file(or captcha), token, site_id, account_id を受け取る
    返却: { ok: True, captcha_url: "https://.../static/captchas/xxx.png" }
    """
    from flask import request, session, jsonify, url_for, current_app
    from pathlib import Path
    from uuid import uuid4
    import time as _time

    # token は /external-seo/start で払い出したもの
    tok = (request.form.get("token") or request.values.get("token") or "").strip()
    if not tok:
        return jsonify({"ok": False, "error": "missing token"}), 400

    # 任意の整合チェック（ズレても致命ではないので警告ログのみ）
    try:
        if session.get("extseo_token") and session["extseo_token"] != tok:
            current_app.logger.warning("[EXTSEO-UP] token mismatch (session present but different)")
    except Exception:
        pass

    # ファイルは 'file' または 'captcha' のどちらでも受ける
    f = request.files.get("file") or request.files.get("captcha")
    if not f or not getattr(f, "filename", ""):
        return jsonify({"ok": False, "error": "no file"}), 400

    # 付帯情報（あれば保存）
    site_id = request.form.get("site_id", type=int)
    account_id = request.form.get("account_id", type=int)

    # 保存
    capt_dir = Path("app/static/captchas")
    capt_dir.mkdir(parents=True, exist_ok=True)
    ts = _time.strftime("%Y%m%d_%H%M%S")
    name = f"captcha_{ts}_{uuid4().hex[:8]}.png"
    save_path = capt_dir / name
    f.save(str(save_path))

    # 公開URL
    captcha_url = url_for("static", filename=f"captchas/{name}", _external=True) + f"?v={int(_time.time())}"

    # ✅ ブラウザセッションに依存せず、トークンで状態を保持
    _extseo_update(tok,
                   step="captcha_shown",
                   progress=20,
                   captcha_url=captcha_url,
                   site_id=site_id,
                   account_id=account_id)

    # 互換: セッションを使うUIが残っている場合のために、入れてもおく（見えない環境なら無視されるだけ）
    try:
        st = dict(session.get("captcha_status") or {})
        st.update({
            "step": "captcha_shown",
            "progress": max(15, int(st.get("progress") or 0)),
            "captcha_url": captcha_url,
            "site_id": site_id or st.get("site_id"),
            "account_id": account_id or st.get("account_id"),
        })
        session["captcha_status"] = st
    except Exception:
        pass

    return jsonify({"ok": True, "captcha_url": captcha_url})


# ===========================
# Topic Anchors / Topic Page
# ===========================

# --- 差分: 新しいルート構成 ---
# 1) /topic/anchors           → アンカー文生成のみ（WP投稿なし）
# 2) /topic/build-skeleton    → WPに下書きを作成（必要なときのみ）
# 3) /topic/generate-now      → クリック瞬間に本文生成・WP更新・URL返却
from urllib.parse import unquote, quote

# =====================================================
# 1️⃣ /topic/anchors （アンカー文生成のみ、骨組み投稿は分離）
# =====================================================
from app.models import Site
@bp.post("/topic/anchors")
def topic_anchors():
    ok, auth_uid = _topic_api_authorized()
    if not ok or not auth_uid:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    site_id = data.get("site_id")
    source_url = data.get("source_url") or ""
    current_title = data.get("current_title") or ""
    page_summary = data.get("page_summary") or ""
    user_traits = data.get("user_traits") or None
    topic_prompt_id = data.get("topic_prompt_id")

    if not site_id:
        return jsonify({"ok": False, "error": "site_id required"}), 400
    site = Site.query.get(site_id)
    if not site:
        return jsonify({"ok": False, "error": "invalid site_id"}), 400
    if auth_uid != site.user_id:
        return jsonify({"ok": False, "error": "forbidden site ownership"}), 403

    # URLが自分のサイト配下か確認（末尾スラッシュ差異を正規化）
    site_base = (site.url or "").rstrip("/")
    if site_base and not source_url.startswith(site_base):
        return jsonify({"ok": False, "error": "invalid source domain"}), 400

    # 1) アンカー文のみ生成
    from app.services.topics import generator as tg
    try:
        anchors = tg.generate_anchor_texts(
            user_id=site.user_id,
            site_id=site_id,
            source_url=source_url,
            current_title=current_title,
            page_summary=page_summary,
            user_traits_json=user_traits,
        )
    except Exception as e:
        current_app.logger.exception("[topic_anchors] anchor generation failed: %s", e)
        return jsonify({"ok": False, "error": "anchor generation failed"}), 500

    # 2) 各アンカーに slug を割り当てる（骨組みはまだ作らない）
    results = {}
    for pos, item in (("top", anchors.top), ("bottom", anchors.bottom)):
        results[pos] = {
            "text": item.text,
            # hrefはgenerate-nowにslug指定で誘導（pos情報を付加）
            "href": url_for(".topic_generate_now", slug=item.slug, pos=pos, _external=True),
        }
    return jsonify({"ok": True, "anchors": results}), 200


# =====================================================
# 2️⃣ /topic/build-skeleton （WP下書きを明示的に作成）
# =====================================================
@bp.post("/topic/build-skeleton")
def topic_build_skeleton():
    ok, auth_uid = _topic_api_authorized()
    if not ok:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    slug = data.get("slug") or ""
    # URLエンコードで渡ってきた slug を DB には常に“生文字列”で保存する
    slug = unquote(slug)
    site_id = data.get("site_id")
    title = data.get("title") or "準備中トピック"
    source_url = data.get("source_url") or ""

    from app.models import TopicPage, Site
    from app.wp_client import post_topic_to_wp
    site = Site.query.get(site_id)
    if not site:
        return jsonify({"ok": False, "error": "invalid site"}), 400
    if auth_uid != site.user_id:
        return jsonify({"ok": False, "error": "forbidden site ownership"}), 403

    page = TopicPage.query.filter_by(slug=slug).first()
    if not page:
        page = TopicPage(
            user_id=site.user_id,
            site_id=site.id,
            slug=slug,
            title=title,
            body="準備中…（数秒後に自動更新されます）",
            meta={"source_url": source_url, "phase": "skeleton"},
        )
        db.session.add(page)
        db.session.commit()

    # WP下書きを投稿（再投稿でも安全）
    # 要件: 自動生成されるtopicページのURLには `topic` を付ける（最小実装として slug に `topic-` を前置）
    wp_slug = slug
    if not wp_slug.startswith("topic-"):
        wp_slug = f"topic-{wp_slug}"
    # ➜ 要望: 自動生成URLに「topic」を付ける（最小実装：slugを topic- 前置）
    wp_slug = slug if slug.startswith("topic-") else f"topic-{slug}"
    post_id, link = post_topic_to_wp(
        site=site,
        title=page.title,
        html="<p>準備中…</p>",
        slug=wp_slug,
    )
    page.meta = dict(page.meta or {}) | {"wp_post_id": post_id}
    page.published_url = link
    db.session.commit()

    return jsonify({"ok": True, "slug": slug, "published_url": link}), 200


# =====================================================
# 3️⃣ /topic/generate-now （クリック瞬間 → 本文生成＆即表示）
# =====================================================
@bp.get("/topic/generate-now")
def topic_generate_now():
    """
    クリック直後に本文を同期生成し、WPを即更新して表示。
    - クエリ: ?slug=<slug>&pos=top|bottom
    """
    from app.models import TopicPage, Site, TopicAnchorLog
    from app.services.topics import generator as tg
    from app.wp_client import update_post_content, post_topic_to_wp
    import time
    # URLから来た slug を“生文字列”に正規化（URLデコード）
    slug = unquote(request.args.get("slug", "") or "")
    pos = request.args.get("pos", "unknown")

    page = TopicPage.query.filter_by(slug=slug).first()
    if not page:
        # 既存データが“URLエンコードslug”で保存されているケースを救済
        encoded = quote(slug, safe="")
        page = TopicPage.query.filter_by(slug=encoded).first()
    if not page:
        abort(404)
    site = Site.query.get(page.site_id)
    if not site:
        abort(404)

    start = time.time()
    # 既に最終生成済みなら即リダイレクト
    if (page.meta or {}).get("phase") == "final" and page.published_url:
        return redirect(page.published_url, code=302)

    # 本文生成
    try:
        affiliates = tg._get_affiliate_links(page.user_id, page.site_id, limit=2)
        filled_prompt = tg.OFFICIAL_TOPIC_PROMPT.format(
            user_traits="{}",
            title="",
            summary="",
            anchor=page.title,
            affiliates=json.dumps(affiliates, ensure_ascii=False),
        )
        out = tg._chat(
            [{"role": "system", "content": "出力形式に厳密に従ってください。"},
             {"role": "user", "content": filled_prompt}],
            max_t=2000, temp=0.5, user_id=page.user_id,
            timeout=0.8  # 明示的な締切（秒）
        )

        m1 = re.search(r"【タイトル】\s*(.+?)\s*【本文】", out, flags=re.DOTALL)
        m2 = re.search(r"【本文】\s*(.+)$", out, flags=re.DOTALL)
        title = (m1.group(1) or page.title).strip() if m1 else page.title
        body = (m2.group(1) or "").strip() if m2 else "本文生成に失敗しました。"

        if affiliates and "おすすめはこちら" not in body:
            a = affiliates[0]
            body += f"\n\nおすすめはこちら：{a.get('title','おすすめ')}（{a.get('url','')}）"

        page.body = body
        page.title = title
        page.meta = dict(page.meta or {}) | {"phase": "final", "gen_ms": int((time.time() - start)*1000)}
        db.session.commit()

        # WP 更新
        html = tg._topic_to_html(page.title, page.body or "")
        post_id = (page.meta or {}).get("wp_post_id")
        if site and post_id:
            update_post_content(site=site, post_id=post_id, new_html=html)
        elif site:
            # 要件: URLに `topic` を付ける
            wp_slug = page.slug if page.slug.startswith("topic-") else f"topic-{page.slug}"
            pid, link = post_topic_to_wp(site=site, title=page.title, html=html, slug=wp_slug)
            page.meta["wp_post_id"] = pid
            page.published_url = link
            db.session.commit()

        # クリックログ
        db.session.add(TopicAnchorLog(
            user_id=page.user_id, site_id=page.site_id, page_id=page.id,
            source_url=(page.meta or {}).get("source_url") or "",
            position=pos, anchor_text=page.title, event="click"
        ))
        db.session.commit()

        # 即リダイレクト
        if page.published_url:
            return redirect(page.published_url, code=302)
        return jsonify({"ok": True, "fallback_used": False, "slug": slug}), 200

    except Exception as e:
        current_app.logger.exception("[topic_generate_now] final generation failed: %s", e)
        # フォールバック（テンプレ本文）
        body = "ページ生成が混み合っています。数秒後に再度お試しください。"
        html = f"<h2>{page.title}</h2><p>{body}</p>"
        post_id = (page.meta or {}).get("wp_post_id")
        if site and post_id:
            update_post_content(site=site, post_id=post_id, new_html=html)
        return jsonify({"ok": True, "fallback_used": True}), 200


# ─────────────────────────────────────────
# リライト機能ユーザー画面（本人専用簡易版）
# ─────────────────────────────────────────
@bp.route("/rewrite", defaults={"username": None}, methods=["GET"])
@bp.route("/<username>/rewrite", methods=["GET"])
@login_required
def user_rewrite_dashboard(username):
    """
    ログインユーザー自身のサイト一覧＋リライト進捗を表示するユーザー用ダッシュボード。
    管理画面 /admin/rewrite/user/<id> と同じ集計ロジックを使う。
    """
    # 管理用と同じヘルパを再利用
    rows = _rewrite_counts_for_user_sites(current_user.id)

    return render_template(
        "rewrite.html",
        user=current_user,
        rows=rows,
    )


@bp.route("/rewrite/enqueue", defaults={"username": None}, methods=["POST"])
@bp.route("/<username>/rewrite/enqueue", methods=["POST"])
@login_required
def user_rewrite_enqueue_self(username):
    payload = request.get_json(silent=True) or {}
    def _to_int_list(v):
        if v is None or v == "":
            return None
        if isinstance(v, list):
            return [int(x) for x in v if str(x).strip().isdigit()]
        return [int(x) for x in str(v).replace("\n", ",").split(",") if x.strip().isdigit()]
    site_ids = _to_int_list(payload.get("site_ids"))
    article_ids = _to_int_list(payload.get("article_ids"))
    priority = float(payload.get("priority", 0.0))
    res = rewrite_enqueue_for_user(current_user.id, site_ids=site_ids, article_ids=article_ids, priority=priority)
    return jsonify({"ok": True, "result": res})

@bp.route("/rewrite/progress", defaults={"username": None}, methods=["GET"])
@bp.route("/<username>/rewrite/progress", methods=["GET"])
@login_required
def user_rewrite_progress_self(username):
    # 管理APIに委譲（user_id 指定）
    with current_app.test_request_context(f"/admin/rewrite/progress?user_id={current_user.id}"):
        return admin_rewrite_progress()

# ─────────────────────────────────────────
# ユーザー用: サイト別のリライト済み記事一覧
# URL:
#   /rewrite/site/<site_id>
#   /<username>/rewrite/site/<site_id>
# ─────────────────────────────────────────
@bp.route("/rewrite/site/<int:site_id>", defaults={"username": None}, methods=["GET"])
@bp.route("/<username>/rewrite/site/<int:site_id>", methods=["GET"])
@login_required
def user_rewrite_site_articles(username, site_id):
    """
    ログインユーザー自身のサイトに対するリライト済み記事一覧。
    管理側 admin_rewrite_site_articles と同じ集計ロジックで、
    HTML も管理テンプレと揃えやすい形に整形する。
    """
    from sqlalchemy import text as _sql
    from urllib.parse import urljoin
    from app.models import Site
    from app.services.rewrite.state_view import fetch_site_totals

    # サイトが current_user 所有かチェック
    site = db.session.get(Site, site_id)
    if not site or site.user_id != current_user.id:
        abort(404)

    # クエリパラメータ
    status = (request.args.get("status") or "").strip().lower()
    page   = max(1, request.args.get("page", type=int) or 1)
    per    = min(100, max(10, request.args.get("per", type=int) or 50))

    # 許容ステータス（success / failed の2系統）
    allowed = {"success", "failed"}
    if status not in allowed:
        status = "success"

    bucket = "success" if status == "success" else "failed"

    # ── サイト全体の統一カウント（ヘッダー用） ─────────────────
    totals = fetch_site_totals(user_id=current_user.id, site_id=site_id)
    stats = {
        "queued":  int(totals.get("waiting", 0)),
        "running": int(totals.get("running", 0)),
        "success": int(totals.get("success", 0)),
        "error":   int(totals.get("failed", 0)),
        "unknown": int(totals.get("other", 0)),
    }
    # 管理テンプレ互換：display_error を必ず持たせる
    stats["display_error"] = stats.get("error", 0)

    # ── 一覧対象 article_id を vw_rewrite_state から抽出（管理側と同じ） ──
    ids_sql = _sql("""
      SELECT article_id
      FROM vw_rewrite_state
      WHERE user_id = :uid AND site_id = :sid AND final_bucket = :bucket
      ORDER BY log_executed_at DESC NULLS LAST,
               plan_created_at DESC NULLS LAST,
               article_id DESC
      LIMIT :limit OFFSET :offset
    """)
    id_rows = db.session.execute(
        ids_sql,
        {
            "uid": current_user.id,
            "sid": site_id,
            "bucket": bucket,
            "limit": per,
            "offset": (page - 1) * per,
        },
    ).fetchall()
    article_ids = [int(r[0]) for r in id_rows]

    # 総件数（ページネーション用）
    total_sql = _sql("""
      SELECT COUNT(*) FROM vw_rewrite_state
      WHERE user_id = :uid AND site_id = :sid AND final_bucket = :bucket
    """)
    total_count = int(
        db.session.execute(
            total_sql,
            {"uid": current_user.id, "sid": site_id, "bucket": bucket},
        ).scalar()
        or 0
    )

    # ── 詳細行を取得（管理側と同じロジック） ────────────────────────
    rows = []
    if article_ids:
        if status == "success":
            # 各記事の最新 success ログ
            detail_sql = _sql("""
              WITH latest AS (
                SELECT
                  l.id         AS log_id,
                  l.article_id,
                  l.plan_id,
                  l.wp_post_id,
                  l.executed_at,
                  ROW_NUMBER() OVER (
                    PARTITION BY l.article_id
                    ORDER BY l.executed_at DESC, l.id DESC
                  ) AS rn
                FROM article_rewrite_logs l
                WHERE l.article_id = ANY(:ids)
                  AND l.wp_status = 'success'
              )
              SELECT
                lt.log_id,
                a.id          AS article_id,
                a.title       AS title,
                lt.plan_id    AS plan_id,
                lt.wp_post_id AS wp_post_id,
                lt.executed_at AS executed_at
              FROM latest lt
              JOIN articles a ON a.id = lt.article_id
              WHERE lt.rn = 1
              ORDER BY lt.executed_at DESC NULLS LAST, a.id DESC
            """)
            rows = list(
                db.session.execute(detail_sql, {"ids": article_ids}).mappings()
            )
        else:
            # 各記事の最新 failed 系ログ
            detail_sql = _sql("""
              WITH latest AS (
                SELECT
                  l.id         AS log_id,
                  l.article_id,
                  l.plan_id,
                  l.wp_post_id,
                  l.executed_at,
                  l.wp_status,
                  ROW_NUMBER() OVER (
                    PARTITION BY l.article_id
                    ORDER BY l.executed_at DESC, l.id DESC
                  ) AS rn
                FROM article_rewrite_logs l
                WHERE l.article_id = ANY(:ids)
                  AND l.wp_status IN (
                    'failed','error','canceled','aborted','timeout','stale'
                  )
              )
              SELECT
                lt.log_id,
                a.id          AS article_id,
                a.title       AS title,
                lt.plan_id    AS plan_id,
                lt.wp_post_id AS wp_post_id,
                lt.executed_at AS executed_at,
                lt.wp_status  AS wp_status
              FROM latest lt
              JOIN articles a ON a.id = lt.article_id
              WHERE lt.rn = 1
              ORDER BY lt.executed_at DESC NULLS LAST, a.id DESC
            """)
            rows = list(
                db.session.execute(detail_sql, {"ids": article_ids}).mappings()
            )

    # ── テンプレ互換: articles 配列を構築（管理テンプレと同じキー構成） ──
    articles = []
    base_url = (site.site_url or site.url or "").rstrip("/")
    _last_dt = None

    for r in rows:
        dt = r.get("executed_at")
        if dt and (_last_dt is None or dt > _last_dt):
            _last_dt = dt

        wp_post_id = r.get("wp_post_id")
        if status == "success" and wp_post_id and base_url:
            wp_url = urljoin(base_url + "/", f"?p={wp_post_id}")
        else:
            wp_url = None

        # 管理テンプレと同じキー名
        articles.append(
            {
                "id":         r.get("article_id"),
                "article_id": r.get("article_id"),
                "title":      r.get("title"),
                "status":     status,  # success / failed
                "updated_at": (dt.isoformat() if dt else None),
                "posted_url": None,
                "wp_url":     wp_url,
                "plan_id":    r.get("plan_id"),
                "log_id":     r.get("log_id"),
            }
        )

    last_updated = _last_dt.isoformat() if _last_dt else None

    # ── ページネーション情報（管理テンプレと同じ構造） ───────────────
    total_pages = (total_count + per - 1) // per if per > 0 else 1
    first_idx = ((page - 1) * per + 1) if total_count > 0 else 0
    last_idx  = min(page * per, total_count)

    prev_url = (
        url_for(
            "main.user_rewrite_site_articles",
            site_id=site_id,
            status=status,
            page=page - 1,
            per=per,
        )
        if page > 1
        else None
    )
    next_url = (
        url_for(
            "main.user_rewrite_site_articles",
            site_id=site_id,
            status=status,
            page=page + 1,
            per=per,
        )
        if page * per < total_count
        else None
    )

    pagination = {
        "total": total_count,
        "page": page,
        "per": per,
        "pages": total_pages,
        "first": first_idx,
        "last": last_idx,
        "prev_url": prev_url,
        "next_url": next_url,
    }

    return render_template(
        "rewrite_site_articles.html",
        site=site,
        site_id=site_id,
        stats=stats,
        status=status,
        per=per,
        articles=articles,
        pagination=pagination,
        last_updated=last_updated,
    )


# ─────────────────────────────────────────
# ユーザー用: リライトログ詳細（修正方針）
# URL:
#   /rewrite/log/<log_id>
#   /<username>/rewrite/log/<log_id>
# ─────────────────────────────────────────
@bp.route("/rewrite/log/<int:log_id>", defaults={"username": None}, methods=["GET"])
@bp.route("/<username>/rewrite/log/<int:log_id>", methods=["GET"])
@login_required
def user_rewrite_log_detail(username, log_id):
    """
    ログID単位の詳細。
    管理画面の修正方針詳細をユーザー向けに簡略表示。
    """
    from urllib.parse import urljoin
    from app.models import ArticleRewriteLog, Article, Site

    log = db.session.get(ArticleRewriteLog, log_id)
    if not log:
        abort(404)

    article = db.session.get(Article, log.article_id)
    if not article or article.user_id != current_user.id:
        # 他人の記事のログは見せない
        abort(404)

    site = db.session.get(Site, article.site_id) if article.site_id else None

    # WPリンク（あくまで簡易。permalink構造は考慮しない）
    wp_url = None
    if site and getattr(log, "wp_post_id", None):
        wp_url = urljoin((site.url or "").rstrip("/") + "/", f"?p={log.wp_post_id}")

    return render_template(
        "rewrite_log_detail.html",
        log=log,
        article=article,
        site=site,
        wp_url=wp_url,
    )

