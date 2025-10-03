# app/utils/redis_lock.py
from __future__ import annotations

"""
軽量な Redis ロックのラッパ。
- SETNX + EXPIRE を使った単純ロック
- コンテキストマネージャで with 句から使用
- 取得待ち（ポーリング）にも対応
"""

import os
import time
import uuid
import random
from typing import Optional

import redis


# 使い回しのためのシングルトン接続
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    url = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
    if not url:
        # シンプルなデフォルト
        url = "redis://127.0.0.1:6379/0"
    if not url.startswith("redis://") and not url.startswith("rediss://"):
        # REDIS_HOST 形式にポートだけ来る運用向け
        host = url
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        _redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        return _redis_client

    _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client


class redis_lock:
    """
    with redis_lock("key", ttl=30, wait=True, wait_timeout=5.0) as acquired:
        if acquired:
            # do work
    """
    def __init__(self, key: str, ttl: int = 30, wait: bool = False, wait_timeout: float = 0.0, poll_interval: float = 0.1):
        self.key = f"lock:{key}"
        self.ttl = max(1, int(ttl))
        self.wait = bool(wait)
        self.wait_timeout = float(wait_timeout)
        self.poll_interval = float(poll_interval)
        self._token: Optional[str] = None
        self._r = _get_redis()

    def acquire_once(self) -> bool:
        token = uuid.uuid4().hex
        try:
            ok = self._r.set(self.key, token, nx=True, ex=self.ttl)
        except Exception:
            # 接続断などは「取得失敗」として扱い、上位でスキップさせる
            ok = False
        if ok:
            self._token = token
            return True
        return False

    def release(self) -> None:
        # 自分のトークンだけ解除（他人のロックを壊さない）
        if not self._token:
            return
        pipe = self._r.pipeline(True)
        while True:
            try:
                pipe.watch(self.key)
                val = pipe.get(self.key)
                if val == self._token:
                    pipe.multi()
                    pipe.delete(self.key)
                    pipe.execute()
                pipe.reset()
                break
            except redis.WatchError:
                pipe.reset()
                break
            except Exception:
                pipe.reset()
                break
        self._token = None

    def __enter__(self) -> bool:
        if not self.wait:
            try:
                return self.acquire_once()
            except Exception:
                return False
        # wait モード：タイムアウトまでポーリング
        deadline = time.time() + max(0.0, self.wait_timeout)
        while True:
            try:
                if self.acquire_once():
                    return True
            except Exception:
                # 取得試行時の一時エラーは待機継続（deadline 管理）
                pass
            if time.time() >= deadline:
                return False
            # 微小ジッタを入れてスパイク回避
            jitter = random.uniform(0.0, max(0.0, self.poll_interval * 0.2))
            time.sleep(max(0.01, self.poll_interval) + jitter)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.release()
        except Exception:
            pass
        return False
