import time
import functools
from sqlalchemy.exc import OperationalError, PendingRollbackError, InterfaceError
from flask import current_app
from app import db

def with_db_retry(max_retries=4, backoff=1.6):
    """
    DB切断やPendingRollback発生時に、自動で rollback→close→dispose→リトライ。
    backoff は指数（1.6^attempt 秒）です。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exc = None
            while attempt <= max_retries:
                try:
                    return fn(*args, **kwargs)
                except (OperationalError, PendingRollbackError, InterfaceError) as e:
                    last_exc = e
                    attempt += 1
                    # 1回目は rollback/close、2回目以降は engine.dispose も併用
                    try:
                        db.session.rollback()
                    except Exception:
                        pass
                    try:
                        db.session.close()
                    except Exception:
                        pass
                    if attempt >= 2:
                        try:
                            # SQLAlchemy 2 でも互換的に取得
                            eng = getattr(db, "engine", None) or getattr(db, "get_engine", lambda: None)()
                            if eng:
                                eng.dispose()
                        except Exception:
                            pass
                    try:
                        current_app.logger.warning(
                            "[db-retry] %s on %s attempt=%s/%s",
                            e.__class__.__name__, getattr(fn, "__name__", str(fn)),
                            attempt, max_retries
                        )
                    except Exception:
                        pass
                    if attempt > max_retries:
                        break
                    # 短めの指数バックオフ（上限クリップ）
                    sleep_s = min(5.0, (backoff ** attempt) * 0.3)
                    time.sleep(sleep_s)
            # ここまで来たら最後の例外を再送
            raise last_exc
        return wrapper
    return deco
