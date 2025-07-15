import logging
from app import db
from app.models import ExternalSEOJobLog

logger = logging.getLogger(__name__)

def log(job_id: int, step: str, message: str):
    try:
        db.session.add(ExternalSEOJobLog(
            job_id=job_id,
            step=step,
            message=message
        ))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.warning(f"[AgentLogger] DB書き込み失敗: {e}")
