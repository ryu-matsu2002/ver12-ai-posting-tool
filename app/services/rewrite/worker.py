# app/services/rewrite/worker.py
# Rewrite専用APSchedulerワーカー（既存機能は呼び出すだけ。既存コードは変更しない）

import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

# ---- アプリ生成の取り出し（どちらの構成でも動くようフォールバック） ----
_app = None
def _get_app():
    global _app
    if _app is not None:
        return _app
    try:
        # 工場関数がある通常構成
        from app import create_app
        _app = create_app()
        return _app
    except Exception:
        # 工場関数が無い場合、wsgiから既存appを拝借
        from wsgi import app as _flask_app
        _app = _flask_app
        return _app

# ---- 既存のrewriteジョブを呼ぶだけ（壊さない） ----
try:
    from app.tasks import rewrite_tick as _rewrite_tick
except Exception:
    _rewrite_tick = None

try:
    from app.tasks import rewrite_retry as _rewrite_retry
except Exception:
    _rewrite_retry = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rewrite_worker")

def _job_rewrite_tick(app):
    if _rewrite_tick is None:
        logger.error("rewrite_tick が見つかりません（app.tasks.rewrite_tick 未定義）")
        return
    with app.app_context():
        _rewrite_tick()

def _job_rewrite_retry(app):
    if _rewrite_retry is None:
        logger.warning("rewrite_retry が未定義のためスキップ（致命的ではありません）")
        return
    with app.app_context():
        _rewrite_retry()

def main():
    # このプロセスはRewrite専用であることを明示（既存コードで参照している可能性に配慮）
    os.environ.setdefault("SCHEDULER_ENABLED", "1")
    os.environ.setdefault("JOBS_ROLE", "rewrite")

    app = _get_app()

    scheduler = BackgroundScheduler(
        executors={
            "default": ThreadPoolExecutor(max_workers=32),
            "rewrite_pool": ThreadPoolExecutor(max_workers=64),
        },
        job_defaults={
            "coalesce": False,
            "max_instances": 4,
            "misfire_grace_time": 60,
        },
        timezone="Asia/Tokyo",
    )

    # 高頻度でrewriteのtickを回す（I/Oが多い想定なのでThreadPoolで十分）
    scheduler.add_job(
        func=_job_rewrite_tick,
        args=[app],
        trigger="interval",
        seconds=5,
        id="rewrite_tick",
        executor="rewrite_pool",
        replace_existing=True,
    )

    # 失敗再試行（定義が無ければ自動的にスキップされる）
    scheduler.add_job(
        func=_job_rewrite_retry,
        args=[app],
        trigger="interval",
        seconds=30,
        id="rewrite_retry",
        executor="rewrite_pool",
        replace_existing=True,
    )

    logger.info("Starting Rewrite Worker scheduler ...")
    scheduler.start()

    try:
        import time
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping Rewrite Worker scheduler ...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()
