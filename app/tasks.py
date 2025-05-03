# app/tasks.py

import logging
from datetime import datetime
import pytz
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler

from . import db
from .models import Article
from .wp_client import post_to_wp  # 統一された WordPress 投稿関数

# グローバルな APScheduler インスタンス（__init__.py で start されています）
scheduler = BackgroundScheduler(timezone="UTC")


def _auto_post_job(app):
    """
    scheduled_at を過ぎた status="done" の記事を
    WordPress に自動投稿し、status="posted" に更新するジョブ。
    """
    with app.app_context():
        now = datetime.now(pytz.utc)  # tz-aware now

        pending = (
            Article.query
                   .filter(Article.status == "done",
                           Article.scheduled_at <= now)
                   .all()
        )

        for art in pending:
            if not art.site:
                current_app.logger.warning(f"記事 {art.id} の投稿先サイト未設定")
                continue

            try:
                url = post_to_wp(art.site, art)
                art.posted_at = now
                art.status = "posted"
                db.session.commit()
                current_app.logger.info(f"Auto-posted Article {art.id} -> {url}")
            except Exception as e:
                current_app.logger.exception(f"Auto-post failed for Article {art.id}: {e}")
                db.session.rollback()

                retry_attempts = 3
                for attempt in range(retry_attempts):
                    try:
                        url = post_to_wp(art.site, art)
                        art.posted_at = now
                        art.status = "posted"
                        db.session.commit()
                        current_app.logger.info(f"Retry Auto-posted Article {art.id} -> {url}")
                        break
                    except Exception as retry_exception:
                        current_app.logger.exception(
                            f"Retry {attempt + 1} failed for Article {art.id}: {retry_exception}"
                        )
                        if attempt == retry_attempts - 1:
                            current_app.logger.error(
                                f"Failed to auto-post Article {art.id} after {retry_attempts} attempts"
                            )


def init_scheduler(app):
    """
    Flask アプリ起動時に呼び出して:
      1) APScheduler に自動投稿ジョブを登録
      2) 5分間隔で _auto_post_job を実行するようスケジュール
    """
    scheduler.add_job(
        func=_auto_post_job,
        trigger="interval",
        minutes=5,
        args=[app],
        id="auto_post_job",
        replace_existing=True,
        max_instances=1
    )
    scheduler.start()
    app.logger.info("Scheduler started: auto_post_job every 5 minutes")
