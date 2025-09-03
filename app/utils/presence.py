# app/utils/presence.py
import os
from datetime import datetime, timezone

try:
    import redis
except Exception:
    redis = None

_PRESENCE_TTL = 90  # 秒
_KEY_FMT = "presence:user:{user_id}"
_r = None

def _get_redis():
    global _r
    if _r is None:
        if redis is None:
            raise RuntimeError("redis-py が未インストールです。`pip install redis` を実行してください。")
        _r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    return _r

def mark_online(user_id: int):
    r = _get_redis()
    r.setex(_KEY_FMT.format(user_id=user_id), _PRESENCE_TTL, "1")

def online_id_set(user_ids):
    r = _get_redis()
    keys = [_KEY_FMT.format(user_id=i) for i in user_ids]
    vals = r.mget(keys) if keys else []
    return {uid for uid, v in zip(user_ids, vals) if v}

def timeago_jp(dt_utc):
    if not dt_utc:
        return "未オンライン"
    now = datetime.now(timezone.utc)
    sec = int((now - dt_utc).total_seconds())
    if sec < 90: return "1分以内"
    mins = sec // 60
    if mins < 60: return f"{mins}分前"
    hours = mins // 60
    if hours < 24: return f"{hours}時間前"
    days = hours // 24
    if days < 7: return f"{days}日前"
    weeks = days // 7
    if weeks < 5: return f"{weeks}週間前"
    months = days // 30
    if months < 12: return f"{months}か月前"
    years = days // 365
    return f"{years}年前"
