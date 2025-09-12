from contextlib import contextmanager
from sqlalchemy import text
from app import db

def _hashkey(key: str) -> int:
    return abs(hash(key)) % (2**31)

@contextmanager
def pg_advisory_lock(key: str):
    k = _hashkey(key)
    db.session.execute(text("SELECT pg_advisory_lock(:k)"), {"k": k})
    try:
        yield
    finally:
        db.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": k})
        db.session.commit()
