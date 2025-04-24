# app/tasks.py

from datetime import datetime
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from flask import current_app
from . import db
from .models import Article
from .wp_client import post_to_wp

def _auto_post_job():
    """
    scheduled_at <= now、status が 'done' の記事を
    WordPress へ自動投稿し、status='posted' に更新。
    """
    now = datetime.utcnow()
    # ① 投稿条件を満たす記事を取得
    to_post = (
        Article.query
        .filter(Article.status == "done",
                Article.site_id.isnot(None),
                Article.scheduled_at <= now)
        .all()
    )

    for art in to_post:
        try:
            url = post_to_wp(art.site, art)
            art.posted_at = now
            art.status    = "posted"
            db.session.commit()
            current_app.logger.info(f"[auto_post] article#{art.id} → {url}")
        except Exception as e:
            current_app.logger.error(f"[auto_post] failed article#{art.id}: {e}")
            db.session.rollback()

def init_scheduler(app):
    """
    アプリ起動時にバックグラウンドスケジューラをセットアップ。
    """
    sched = BackgroundScheduler(daemon=True)
    # １分おきに _auto_post_job を呼び出し
    sched.add_job(
        func=_auto_post_job,
        trigger="interval",
        minutes=1,
        id="auto_post_job",
        replace_existing=True
    )
    sched.start()
    app.logger.info("Scheduler for auto-post started (interval: 1 min).")
