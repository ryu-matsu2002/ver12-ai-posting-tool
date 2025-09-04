# app/utils/cache.py
import json, os
import redis

_redis = None

def get_redis():
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis = redis.from_url(url, decode_responses=True)
    return _redis

def cache_get_json(key):
    r = get_redis()
    v = r.get(key)
    return json.loads(v) if v else None

def cache_set_json(key, value, ttl=300):
    r = get_redis()
    r.setex(key, ttl, json.dumps(value))
