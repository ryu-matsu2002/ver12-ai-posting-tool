# app/services/internal_seo/enqueue.py
import os
from sqlalchemy import text
from app import db

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
    db.session.execute(text("""
        INSERT INTO internal_seo_job_queue
          (site_id, pages, per_page, min_score, max_k, limit_sources, limit_posts,
           incremental, job_kind, status, created_at)
        VALUES
          (:site_id, :pages, :per_page, :min_score, :max_k, :limit_sources, :limit_posts,
           :incremental, :job_kind, 'queued', now())
    """), params)
    db.session.commit()

# --- appended: provide enqueue_new_site API for internal SEO ---

from __future__ import annotations
import os
from typing import Optional, Dict, Any
from sqlalchemy import text
try:
    from flask import current_app  # ログ出力に使う（コンテキスト外でも安全に）
except Exception:
    current_app = None  # type: ignore

from app import db
from app.models import Site

__all__ = ["enqueue_new_site"]

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
