# app/services/rewrite/worker.py
# Rewrite専用APSchedulerワーカー（既存機能は呼び出すだけ。既存コードは変更しない）

import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
import importlib
import logging

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

# フォールバック解決のキャッシュ
_tick_fn_cache = None
_retry_fn_cache = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rewrite_worker")

# ─────────────────────────────────────────
# rewrite_tick / rewrite_retry の動的解決
# 優先順:
#   1) app.tasks
#   2) app.services.rewrite.bulk_runner
#   3) app.services.rewrite.executor
# どこにも無ければログ出してスキップ（致命ではない）
# ─────────────────────────────────────────
def _resolve_funcs():
    candidates = [
        ("app.tasks", "rewrite_tick", "rewrite_retry"),
        ("app.services.rewrite.bulk_runner", "rewrite_tick", "rewrite_retry"),
        ("app.services.rewrite.executor", "rewrite_tick", "rewrite_retry"),
    ]
    tick_fn = retry_fn = None
    for mod_name, tick_name, retry_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if tick_fn is None and hasattr(mod, tick_name):
                tick_fn = getattr(mod, tick_name)
                logger.info("rewrite_tick を %s から解決", mod_name)
            if retry_fn is None and hasattr(mod, retry_name):
                retry_fn = getattr(mod, retry_name)
                logger.info("rewrite_retry を %s から解決", mod_name)
            if tick_fn and retry_fn:
                break
        except Exception as e:
            logger.debug("func resolve skipped (%s): %s", mod_name, e)
    return tick_fn, retry_fn

def _job_rewrite_tick(app):
    global _tick_fn_cache
    fn = _rewrite_tick
    if fn is None:
        if _tick_fn_cache is None:
            tick_fn, _ = _resolve_funcs()
            _tick_fn_cache = tick_fn
        fn = _tick_fn_cache
    if fn is None:
        # bulk_runner には rewrite_tick_once があるので、それも探す
        try:
            br = importlib.import_module("app.services.rewrite.bulk_runner")
            if hasattr(br, "rewrite_tick_once"):
                fn = getattr(br, "rewrite_tick_once")
                _tick_fn_cache = fn
                logger.info("rewrite_tick を bulk_runner.rewrite_tick_once へフォールバック")
        except Exception as e:
            logger.debug("bulk_runner fallback failed: %s", e)
    if fn is None:
        logger.error("rewrite_tick が見つかりません（app.tasks もフォールバックも未定義）")
        return
    with app.app_context():
        # 引数違いに対応（fn(app, ...) or fn()）
        try:
            return fn(app, dry_run=None)
        except TypeError:
            return fn()

def _job_rewrite_retry(app):
    global _retry_fn_cache
    fn = _rewrite_retry
    if fn is None:
        if _retry_fn_cache is None:
            _, retry_fn = _resolve_funcs()
            _retry_fn_cache = retry_fn
        fn = _retry_fn_cache
    if fn is None:
        # bulk_runner の失敗再試行APIにフォールバック
        try:
            br = importlib.import_module("app.services.rewrite.bulk_runner")
            if hasattr(br, "retry_failed_plans"):
                fn = getattr(br, "retry_failed_plans")
                _retry_fn_cache = fn
                logger.info("rewrite_retry を bulk_runner.retry_failed_plans へフォールバック")
        except Exception as e:
            logger.debug("bulk_runner retry fallback failed: %s", e)
    if fn is None:
        logger.warning("rewrite_retry が未定義のためスキップ（致命的ではありません）")
        return
    with app.app_context():
        try:
            return fn(app)
        except TypeError:
            return fn()


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

    # ─────────────────────────────────────────────
    # REWRITE_SLOT 環境変数でジョブIDを一意化
    # 同一IDだと複数ワーカーが競合し、skip連発・置換事故が起きるため
    # ─────────────────────────────────────────────
    from datetime import datetime, timezone
    slot = os.environ.get("REWRITE_SLOT", "0")
    tick_job_id = f"rewrite_tick:{slot}"
    retry_job_id = f"rewrite_retry:{slot}"

    # rewrite_tick（5秒ごと）
    scheduler.add_job(
        func=_job_rewrite_tick,
        args=[app],
        trigger="interval",
        seconds=5,
        id=tick_job_id,
        executor="rewrite_pool",
        max_instances=1,            # 同一ワーカー内で重複実行防止
        coalesce=True,              # 遅延時は1回に圧縮
        replace_existing=True,      # 再起動時に置換
        misfire_grace_time=10,      # 軽い遅延を許容
        next_run_time=datetime.now(timezone.utc),  # 即初回起動
    )

    # rewrite_retry（30秒ごと）
    scheduler.add_job(
        func=_job_rewrite_retry,
        args=[app],
        trigger="interval",
        seconds=30,
        id=retry_job_id,
        executor="rewrite_pool",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
        misfire_grace_time=30,
        next_run_time=datetime.now(timezone.utc),
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
