from datetime import datetime
from pytz import utc

from . import celery, db
from .models import Article
from .wp_client import post_to_wp

@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **_):
    # 60 秒ごとに未投稿記事をチェック
    sender.add_periodic_task(60, schedule_pending_jobs.s())

@celery.task
def schedule_pending_jobs():
    now = datetime.utcnow().replace(tzinfo=utc)
    q = (Article.query
         .filter(Article.status == "done",
                 Article.scheduled_at <= now,
                 Article.posted_at.is_(None),
                 Article.site_id.isnot(None)))
    for art in q.all():
        try:
            url = post_to_wp(art.site, art)
            art.status, art.posted_at = "posted", datetime.utcnow()
            db.session.commit()
        except Exception as e:
            art.status = "error"
            art.body   = f"WP 投稿失敗: {e}"
            db.session.commit()
