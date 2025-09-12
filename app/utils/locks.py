from contextlib import contextmanager
from sqlalchemy import text
from app import db
import hashlib
import logging

logger = logging.getLogger(__name__)

def _hashkey(key: str) -> int:
    """
    プロセス間で安定するロックキーを生成する。
    Pythonのbuilt-in hash() はプロセスごとに乱数シードが異なるため使用しない。
    ここでは MD5 の先頭32bit を取り、符号付き31bitに収める。
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    # 先頭8桁(32bit) → 0x7FFFFFFF で31bitにクリップ（pg_advisory_lock は BIGINT 受け取り可）
    return int(h[:8], 16) & 0x7FFFFFFF

@contextmanager
def pg_advisory_lock(key: str):
    """
    PostgreSQL セッションレベルのアドバイザリロック。
    - 同じ key に対しては全プロセス/全スレッドで直列化される。
    - ブロッキングで取得（解放まで待つ）。
    """
    k = _hashkey(key)
    # 取得
    db.session.execute(text("SELECT pg_advisory_lock(:k)"), {"k": k})
    logger.info("[locks] acquired pg_advisory_lock key=%s (k=%d)", key, k)
    try:
        yield
    finally:
        # 解放してコミット（同一SQLAlchemyセッションの接続で実行される想定）
        db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": k})
        db.session.commit()
        logger.info("[locks] released pg_advisory_lock key=%s (k=%d)", key, k)
