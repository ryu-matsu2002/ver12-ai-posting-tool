# app/tasks.py

import logging
from datetime import datetime
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler

from . import db
from .models import Article
from .wp_client import post_to_wp  # WordPress 投稿ユーティリティ

# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")


def _auto_post_job(app):
    """scheduled_at を過ぎた記事を WordPress に自動投稿し、ステータスを更新するジョブ"""
    with app.app_context():
        now = datetime.utcnow()
        # ステータス "done"（生成済み）のうち scheduled_at <= now のものを取得
        pending = (
            Article.query
                   .filter(Article.scheduled_at <= now, Article.status == "done")
                   .all()
        )

        for art in pending:
            try:
                url = post_to_wp(art.site, art)
                art.posted_at = now
                art.status    = "posted"
                db.session.commit()
                current_app.logger.info(f"Auto-posted Article {art.id} -> {url}")
            except Exception:
                current_app.logger.exception(f"Auto-post failed for Article {art.id}")
                art.status = "error"
                db.session.commit()


def init_scheduler(app):
    """
    アプリ起動時に呼び出して、
    1) APScheduler に自動投稿ジョブを登録
    2) 1 分間隔で _auto_post_job を実行するようスケジュール
    """
    # 既存ジョブがあれば置き換え
    scheduler.add_job(
        func=_auto_post_job,
        trigger="interval",
        minutes=1,
        args=[app],
        id="auto_post_job",
        replace_existing=True
    )
    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 1 minute")
