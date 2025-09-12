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
