# app/tasks/title_meta_backfill.py

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app import db
from app.models import Article
from flask import current_app

# --- ユーティリティ ----------------------------------------------------------

def _now():
    # timezone=True のカラムに合わせて naive UTC を避ける（DB側でUTC管理）
    return datetime.utcnow()

def _coerce_limit(v: Optional[int], default: int = 200, max_cap: int = 1000) -> int:
    try:
        i = int(v or 0)
    except Exception:
        i = 0
    if i <= 0:
        i = default
    return max(1, min(i, max_cap))

def _as_bool(v: Any, default: bool=False) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1","true","yes","on"):
        return True
    if s in ("0","false","no","off"):
        return False
    return default

# --- メタ説明の生成ロジック（最小実装） --------------------------------------
# ここはあなたの本番生成器に差し替え可。
# ない場合でも“最低限の動作”として本文冒頭からサマライズする。
def _make_meta_description(title: str, body: Optional[str], limit_chars: int = 180) -> str:
    if (body or "").strip():
        text = body.strip().replace("\r", " ").replace("\n", " ")
        text = " ".join(text.split())
        cand = text[: limit_chars].strip()
        if len(cand) < 40 and title:
            cand = f"{title} – {cand}"
        return cand[: limit_chars]
    # body が空ならタイトルで埋める（WP側で上書き可）
    return (title or "")[: limit_chars]

def _judge_quality(meta_desc: str) -> str:
    l = len(meta_desc or "")
    if l == 0:
        return "empty"
    if l < 60:
        return "too_short"
    if l > 180:
        return "too_long"
    return "ok"

# --- WP 反映（フックだけ。既存のあなたの実装に差し替え可能） ---------------
def _push_meta_to_wp_if_needed(article: Article) -> None:
    """
    WP反映ロジックに接続するフック。
    既存プロジェクトに WP クライアントがあるならここで呼ぶ。
    何もなければ安全に return。
    """
    try:
        # 例:
        # from app.services.wp_sync import update_meta_description
        # if article.wp_post_id and article.posted_url:
        #     update_meta_description(site=article.site, wp_post_id=article.wp_post_id, meta=article.meta_description)
        return
    except Exception as e:
        current_app.logger.exception("[title-meta] WP push failed article_id=%s err=%s", article.id, e)

# --- 本体 --------------------------------------------------------------------

def run_title_meta_backfill(
    site_id: Optional[int] = None,
    user_id: Optional[int] = None,
    limit: Optional[int] = 200,
    dryrun: Optional[bool] = False,
    after_id: Optional[int] = None,
    push_to_wp: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Title/Meta バックフィル実行。
    - フィルタ: is_manual_meta=False AND status IN ('done','posted')
             AND meta_desc_quality IN ('empty','too_short','too_long','duplicate')
    - カーソル: id > after_id の昇順
    - 返却: {ok, updated, cursor, done}
    """
    dryrun = _as_bool(dryrun, False)
    push_to_wp = _as_bool(push_to_wp, False)
    limit_int = _coerce_limit(limit, default=200, max_cap=1000)

    qualities = ("empty", "too_short", "too_long", "duplicate")
    statuses  = ("done", "posted")

    q = (
        db.session.query(Article)
        .filter(Article.is_manual_meta == False)  # noqa: E712
        .filter(Article.status.in_(statuses))
        .filter(Article.meta_desc_quality.in_(qualities))
    )

    if user_id:
        q = q.filter(Article.user_id == int(user_id))
    if site_id:
        q = q.filter(Article.site_id == int(site_id))
    if after_id:
        q = q.filter(Article.id > int(after_id))

    q = q.order_by(Article.id.asc()).limit(limit_int)

    rows = q.all()
    if not rows:
        return {"ok": True, "updated": 0, "cursor": after_id, "done": True}

    updated = 0
    last_id = after_id or 0

    for art in rows:
        last_id = art.id
        # 既存メタを再評価（short/long/dup なども再生成）
        new_meta = _make_meta_description(art.title or "", art.body or "", limit_chars=180)
        new_quality = _judge_quality(new_meta)

        # dryrun ならDB変更しない
        if dryrun:
            continue

        # 変更がある時だけ更新
        changed = False
        if (art.meta_description or "") != new_meta:
            art.meta_description = new_meta
            changed = True

        if (art.meta_desc_quality or "") != new_quality:
            art.meta_desc_quality = new_quality
            changed = True

        if changed:
            art.meta_desc_last_updated_at = _now()
            updated += 1

        # WP反映（posted のみ）。重くなるので try/except で個別に処理
        if push_to_wp and art.status == "posted":
            _push_meta_to_wp_if_needed(art)

    if not dryrun:
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception("[title-meta] commit failed after_id=%s err=%s", after_id, e)
            return {"ok": False, "error": f"commit failed: {e}"}

    done = (len(rows) < limit_int)
    return {
        "ok": True,
        "updated": updated,
        "cursor": last_id,
        "done": done,
    }
