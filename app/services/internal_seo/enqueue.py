# app/services/internal_seo/enqueue.py
import os
from typing import Optional, Dict, Any
from sqlalchemy import text
from app import db
from app.models import Site, InternalSeoUserSchedule
try:
    from flask import current_app  # ログ出力に使う（任意）
except Exception:
    current_app = None  # type: ignore

__all__ = [
    "enqueue_new_site",
    "enqueue_internal_seo_for_site",
    "enqueue_refill_for_site",            # 🆕 追加：弾補給（適用なし）を投入
]

def enqueue_internal_seo_for_site(site_id: int, kind: str = "new-site") -> None:
    """
    指定サイトを internal_seo_job_queue に即時投入する。
    ENV のデフォルト値をそのまま用いる（管理画面の一括投入でも再利用可）
    """
    params = {
        "site_id": site_id,
        "pages": int(os.getenv("INTERNAL_SEO_PAGES", 10)),
        "per_page": int(os.getenv("INTERNAL_SEO_PER_PAGE", 100)),
        "min_score": float(os.getenv("INTERNAL_SEO_MIN_SCORE", 0.05)),
        "max_k": int(os.getenv("INTERNAL_SEO_MAX_K", 80)),
        "limit_sources": int(os.getenv("INTERNAL_SEO_LIMIT_SOURCES", 200)),
        "limit_posts": int(os.getenv("INTERNAL_SEO_LIMIT_POSTS", 50)),
        "incremental": os.getenv("INTERNAL_SEO_INCREMENTAL", "1") == "1",
        "job_kind": kind,
    }

    # 既存 queued/running があれば重複投入しない
    exists = db.session.execute(text("""
        SELECT 1
          FROM internal_seo_job_queue
         WHERE site_id = :sid
           AND status IN ('queued','running')
         LIMIT 1
    """), {"sid": site_id}).first()
    if exists:
        return

    db.session.execute(text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        VALUES
          (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
           :incremental, :job_kind, 'queued', now())
    """), params)
    db.session.commit()

# --- provide enqueue_new_site API for internal SEO ---

def enqueue_new_site(site_id: int,
                     pages: Optional[int] = None,
                     per_page: Optional[int] = None,
                     min_score: Optional[float] = None,
                     max_k: Optional[int] = None,
                     limit_sources: Optional[int] = None,
                     limit_posts: Optional[int] = None,
                     incremental: Optional[bool] = None,
                     job_kind: str = "auto-enqueue") -> Dict[str, Any]:
    """
    新規サイト登録時などに内部SEOジョブをキューへ投入する軽量API。

    仕様:
      - サイトが存在しない場合: {"ok": False, "enqueued": False, "reason": "site-not-found"}
      - すでに queued/running のジョブがある場合: {"ok": True, "enqueued": False, "reason": "already-queued-or-running"}
      - それ以外は INSERT して {"ok": True, "enqueued": True, "reason": "inserted"}

    ENVデフォルト（未指定引数に適用）:
      INTERNAL_SEO_PAGES=10
      INTERNAL_SEO_PER_PAGE=100
      INTERNAL_SEO_MIN_SCORE=0.05
      INTERNAL_SEO_MAX_K=80
      INTERNAL_SEO_LIMIT_SOURCES=200
      INTERNAL_SEO_LIMIT_POSTS=50
      INTERNAL_SEO_INCREMENTAL=1  (真偽)
    """
    # サイト存在チェック
    site = Site.query.get(site_id)
    if not site:
        return {"ok": False, "enqueued": False, "reason": "site-not-found"}
    
    # 🛡 ユーザー単位スケジュールの有効性チェック
    sched = InternalSeoUserSchedule.query.filter_by(user_id=site.user_id).one_or_none()
    if not sched:
        return {"ok": False, "enqueued": False, "reason": "user-schedule-missing"}
    if not sched.is_enabled:
        return {"ok": True, "enqueued": False, "reason": "user-schedule-disabled"}
    if getattr(sched, "status", None) == "paused":
        return {"ok": True, "enqueued": False, "reason": "user-schedule-paused"}

    # 既存の queued/running を確認（重複投入の防止）
    exists = db.session.execute(text("""
        SELECT 1
          FROM internal_seo_job_queue
         WHERE site_id = :sid
           AND status IN ('queued','running')
         LIMIT 1
    """), {"sid": site_id}).first()
    if exists:
        return {"ok": True, "enqueued": False, "reason": "already-queued-or-running"}

    # デフォルト値（環境変数優先）
    def _env_int(k: str, d: int) -> int:
        try:
            return int(os.getenv(k, d))
        except Exception:
            return d

    def _env_float(k: str, d: float) -> float:
        try:
            return float(os.getenv(k, d))
        except Exception:
            return d

    def _env_bool(k: str, d: bool) -> bool:
        v = os.getenv(k)
        if v is None:
            return d
        return v.lower() in ("1", "true", "yes", "y", "on")

    pages         = pages         if pages         is not None else _env_int("INTERNAL_SEO_PAGES", 10)
    per_page      = per_page      if per_page      is not None else _env_int("INTERNAL_SEO_PER_PAGE", 100)
    min_score     = min_score     if min_score     is not None else _env_float("INTERNAL_SEO_MIN_SCORE", 0.05)
    max_k         = max_k         if max_k         is not None else _env_int("INTERNAL_SEO_MAX_K", 80)
    limit_sources = limit_sources if limit_sources is not None else _env_int("INTERNAL_SEO_LIMIT_SOURCES", 200)
    limit_posts   = limit_posts   if limit_posts   is not None else _env_int("INTERNAL_SEO_LIMIT_POSTS", 50)
    incremental   = incremental   if incremental   is not None else _env_bool("INTERNAL_SEO_INCREMENTAL", True)

    # キュー投入
    db.session.execute(text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        VALUES
          (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
           :incremental, :job_kind, 'queued', now())
    """), {
        "site_id": site_id,
        "pages": pages,
        "per_page": per_page,
        "min_score": min_score,
        "max_k": max_k,
        "limit_sources": limit_sources,
        "limit_posts": limit_posts,
        "incremental": bool(incremental),
        "job_kind": job_kind,
    })
    db.session.commit()

    try:
        if current_app:
            current_app.logger.info("[internal-seo enqueue] site=%s kind=%s", site_id, job_kind)
    except Exception:
        # ロガー無しでも処理は継続
        pass

    return {"ok": True, "enqueued": True, "reason": "inserted"}


# ----------------------------------------------------------------------
# 🆕 軽量：refill（弾補給）用キュー投入
#   - 目的：indexer + planner だけを実行し、pending を補充する
#   - 実装：limit_posts=0 を明示してジョブ投入（applier はスキップされる）
#   - 既存 queued/running があれば重複投入しない（他ジョブと同じ挙動）
#   - ENV（存在すれば優先）:
#       INTERNAL_SEO_REFILL_PAGES
#       INTERNAL_SEO_REFILL_PER_PAGE
#       INTERNAL_SEO_REFILL_MIN_SCORE
#       INTERNAL_SEO_REFILL_MAX_K
#       INTERNAL_SEO_REFILL_LIMIT_SOURCES
#       （limit_posts は常に 0 固定）
# ----------------------------------------------------------------------
def enqueue_refill_for_site(
    site_id: int,
    *,
    pages: Optional[int] = None,
    per_page: Optional[int] = None,
    min_score: Optional[float] = None,
    max_k: Optional[int] = None,
    limit_sources: Optional[int] = None,
    incremental: Optional[bool] = None,
    job_kind: str = "refill-plan-only",
) -> Dict[str, Any]:
    """
    弾補給（indexer+planner のみ）を internal_seo_job_queue に投入する。
    limit_posts=0 で投入するため、ワーカーは適用フェーズ（applier）を実行しない。

    返り値:
      {"ok": True/False, "enqueued": True/False, "reason": "..."}
    """
    # サイト存在チェック（存在しないなら何もしない）
    site = Site.query.get(site_id)
    if not site:
        return {"ok": False, "enqueued": False, "reason": "site-not-found"}

    # 既存 queued/running があれば投入しない（過負荷防止）
    exists = db.session.execute(text("""
        SELECT 1
          FROM internal_seo_job_queue
         WHERE site_id = :sid
           AND status IN ('queued','running')
         LIMIT 1
    """), {"sid": site_id}).first()
    if exists:
        return {"ok": True, "enqueued": False, "reason": "already-queued-or-running"}

    # ENV優先のデフォルト解決（refill専用ENV → 通常ENV → ハードデフォルト）
    def _env_int(keys, d: int) -> int:
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        return d

    def _env_float(keys, d: float) -> float:
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return d

    def _env_bool(keys, d: bool) -> bool:
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                return v.lower() in ("1", "true", "yes", "y", "on")
        return d

    pages = (
        pages if pages is not None else
        _env_int(["INTERNAL_SEO_REFILL_PAGES", "INTERNAL_SEO_PAGES"], 5)
    )
    per_page = (
        per_page if per_page is not None else
        _env_int(["INTERNAL_SEO_REFILL_PER_PAGE", "INTERNAL_SEO_PER_PAGE"], 100)
    )
    min_score = (
        min_score if min_score is not None else
        _env_float(["INTERNAL_SEO_REFILL_MIN_SCORE", "INTERNAL_SEO_MIN_SCORE"], 0.05)
    )
    max_k = (
        max_k if max_k is not None else
        _env_int(["INTERNAL_SEO_REFILL_MAX_K", "INTERNAL_SEO_MAX_K"], 80)
    )
    limit_sources = (
        limit_sources if limit_sources is not None else
        _env_int(["INTERNAL_SEO_REFILL_LIMIT_SOURCES", "INTERNAL_SEO_LIMIT_SOURCES"], 200)
    )
    incremental = (
        incremental if incremental is not None else
        _env_bool(["INTERNAL_SEO_REFILL_INCREMENTAL", "INTERNAL_SEO_INCREMENTAL"], True)
    )

    params = {
        "site_id": site_id,
        "pages": int(pages),
        "per_page": int(per_page),
        "min_score": float(min_score),
        "max_k": int(max_k),
        "limit_sources": int(limit_sources),
        "limit_posts": 0,                       # ★ ここがポイント：適用フェーズを実行させない
        "incremental": bool(incremental),
        "job_kind": job_kind,                   # 追跡しやすい kind
    }

    db.session.execute(text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        VALUES
          (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
           :incremental, :job_kind, 'queued', now())
    """), params)
    db.session.commit()

    try:
        if current_app:
            current_app.logger.info(
                "[internal-seo enqueue REFILL] user=%s site=%s kind=%s params=%s",
                site.user_id, site_id, job_kind, {k: v for k, v in params.items() if k != "site_id"}
            )
    except Exception:
        pass

    return {"ok": True, "enqueued": True, "reason": "inserted"}