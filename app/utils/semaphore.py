# app/utils/semaphore.py
import os, time, uuid
from contextlib import contextmanager
from typing import Optional

LIMIT = int(os.getenv("EXTSEO_CONCURRENCY_LIMIT", "3"))
KEY   = os.getenv("EXTSEO_SEMAPHORE_KEY", "extseo:semaphore")
TTL   = int(os.getenv("EXTSEO_SEMAPHORE_TTL", "600"))   # 10分で自動解放

def _get_redis():
    from redis import Redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(url, decode_responses=True)

def current_active() -> int:
    r = _get_redis()
    # 有効なトークンだけ数える
    tokens = r.smembers(KEY)
    valid = 0
    for t in tokens:
        if r.exists(f"{KEY}:{t}"):
            valid += 1
        else:
            # TTL切れで個別キー消えた → セットからも掃除
            r.srem(KEY, t)
    return valid

def try_acquire() -> Optional[str]:
    """
    空きがあればトークン(=メンバーID)を追加して確保、満杯ならNone。
    """
    r = _get_redis()
    if current_active() >= LIMIT:
        return None
    token = str(uuid.uuid4())
    pipe = r.pipeline()
    pipe.sadd(KEY, token)
    # 個別キーにTTLを付与（10分後に自動削除される）
    pipe.setex(f"{KEY}:{token}", TTL, "1")
    pipe.execute()
    return token

def release(token: str):
    r = _get_redis()
    r.srem(KEY, token)
    r.delete(f"{KEY}:{token}")

@contextmanager
def guard():
    token = try_acquire()
    if not token:
        yield None
        return
    try:
        yield token
    finally:
        release(token)
