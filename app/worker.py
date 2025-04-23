# worker.py  ─ Celery ワーカー (予約投稿)
# 実行コマンド例:  celery -A worker.celery worker --loglevel=info
#─────────────────────────────────────────────
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

from app import create_app, make_celery, db
from app.models import Article
from app.wp_client import post_to_wp

# ─── Flask & Celery インスタンス生成 ──────────
flask_app = create_app()
celery = make_celery(flask_app)


# ------------------------------------------------
# 定期タスク: 60 秒ごとにスケジュール投稿を実行
# ------------------------------------------------
@celery.on_after_configure.connect
def _setup_periodic_tasks(sender, **_):
    # 60 秒周期 (設定は環境に合わせて調整可)
    sender.add_periodic_task(60.0, post_scheduled.s(), name="check‐auto‐post")


@celery.task
def post_scheduled():
    """scheduled_at <= 現在 の記事を WordPress へ投稿"""
    with flask_app.app_context():
        now = datetime.now(timezone.utc)
        q = (
            Article.query.filter(
                Article.status == "done",
                Article.posted_at.is_(None),
                Article.scheduled_at.isnot(None),
                Article.scheduled_at <= now,
            )
            .order_by(Article.scheduled_at.asc())
        )

        for art in q.all():
            try:
                wp_link = post_to_wp(art.site, art)  # 投稿
                art.posted_at = datetime.now(timezone.utc)
                db.session.commit()
                logging.info("Auto-posted Article[%s] -> %s", art.id, wp_link)
            except Exception as e:
                # エラーはロールバックし、次回リトライ対象に残す
                db.session.rollback()
                logging.exception("WP 投稿失敗 Article[%s]: %s", art.id, e)
