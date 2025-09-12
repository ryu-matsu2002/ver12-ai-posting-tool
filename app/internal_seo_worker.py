# app/internal_seo_worker.py
import os
import time
import signal
import logging
from contextlib import contextmanager

from sqlalchemy import text
from app import create_app, db
from app.tasks import _internal_seo_run_one  # 既存の実行関数を呼び出す

LOG = logging.getLogger("internal_seo_worker")
logging.basicConfig(
    level=os.getenv("INTERNAL_SEO_WORKER_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ループ間隔（秒）
POLL_INTERVAL_SEC = float(os.getenv("INTERNAL_SEO_WORKER_INTERVAL", "2.0"))
# 連続エラー時のリトライ待機（秒）
BACKOFF_SEC = float(os.getenv("INTERNAL_SEO_WORKER_BACKOFF", "5.0"))

_shutdown = False


def _setup_signals():
    def _handler(sig, frame):
        global _shutdown
        LOG.info(f"signal received: {sig}, preparing to shutdown…")
        _shutdown = True

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _handler)


@contextmanager
def app_context():
    app = create_app()
    with app.app_context():
        yield


def _pick_one_queued_job():
    """
    キューから1件だけ安全にピックアップ。
    SELECT … FOR UPDATE SKIP LOCKED で多重実行を防ぎつつ並列にも対応可能。
    """
    with db.session.begin():
        row = db.session.execute(text("""
            SELECT id
              FROM internal_seo_job_queue
             WHERE status = 'queued'
             ORDER BY created_at ASC
             FOR UPDATE SKIP LOCKED
             LIMIT 1
        """)).first()

        if not row:
            return None

        job_id = row[0]
        db.session.execute(text("""
            UPDATE internal_seo_job_queue
               SET status='running', started_at=now()
             WHERE id=:id
        """), {"id": job_id})

    job = db.session.execute(text("""
        SELECT *
          FROM internal_seo_job_queue
         WHERE id=:id
    """), {"id": job_id}).mappings().first()
    return job


def _mark_done(job_id: int):
    with db.session.begin():
        db.session.execute(text("""
            UPDATE internal_seo_job_queue
               SET status='done', ended_at=now()
             WHERE id=:id
        """), {"id": job_id})


def _mark_error(job_id: int, err: str):
    err = (err or "")[:10000]
    with db.session.begin():
        db.session.execute(text("""
            UPDATE internal_seo_job_queue
               SET status='error', message=:err, ended_at=now()
             WHERE id=:id
        """), {"id": job_id, "err": err})


def worker_loop():
    LOG.info("Internal SEO worker started.")
    _setup_signals()

    while not _shutdown:
        try:
            job = _pick_one_queued_job()
            if not job:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            job_id = job["id"]
            LOG.info(f"[job:{job_id}] start site_id={job['site_id']} kind={job['job_kind']}")

            # 実行本体：既存の関数をそのまま呼ぶ
            _internal_seo_run_one(
                site_id=job["site_id"],
                pages=job["pages"],
                per_page=job["per_page"],
                min_score=job["min_score"],
                max_k=job["max_k"],
                limit_sources=job["limit_sources"],
                limit_posts=job["limit_posts"],
                incremental=job["incremental"],
                job_kind=job["job_kind"],
            )

            _mark_done(job_id)
            LOG.info(f"[job:{job_id}] done")

        except Exception as e:
            LOG.exception("worker loop error")
            # job_id が取れていれば error 反映
            try:
                if "job_id" in locals():
                    _mark_error(job_id, str(e))
            except Exception:
                LOG.exception("failed to mark job error")
            time.sleep(BACKOFF_SEC)

    LOG.info("Internal SEO worker stopped (graceful).")


if __name__ == "__main__":
    with app_context():
        worker_loop()
