import time
import functools
from sqlalchemy.exc import OperationalError, PendingRollbackError
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
            while True:
                try:
                    return fn(*args, **kwargs)
                except (OperationalError, PendingRollbackError):
                    attempt += 1
                    db.session.rollback()
                    db.session.close()
                    db.engine.dispose()
                    if attempt > max_retries:
                        raise
                    time.sleep(backoff ** attempt)
        return wrapper
    return deco
