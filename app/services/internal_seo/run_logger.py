# app/services/internal_seo/run_logger.py
from datetime import datetime
from typing import Optional, Dict, Any
from app import db
from app.models import InternalSeoRun

class RunLogger:
    def __init__(self, site_id: int, job_kind: str = "manual"):
        self.site_id = site_id
        self.job_kind = job_kind
        self.run: Optional[InternalSeoRun] = None
        self._started_at: Optional[datetime] = None

    def start(self, params: Optional[Dict[str, Any]] = None) -> InternalSeoRun:
        self._started_at = datetime.utcnow()
        self.run = InternalSeoRun(
            site_id=self.site_id,
            job_kind=self.job_kind,
            status="running",
            started_at=self._started_at,
            stats={"params": params or {}},
        )
        db.session.add(self.run)
        db.session.commit()
        return self.run

    def update_stats(self, patch: Dict[str, Any]) -> None:
        if not self.run:
            return
        stats = self.run.stats or {}
        # 浅いマージ（必要ならdeepmergeに置き換え可能）
        for k, v in patch.items():
            stats[k] = v
        self.run.stats = stats
        db.session.commit()

    def finish(self, status: str, extra_stats: Optional[Dict[str, Any]] = None, message: str = "") -> None:
        if not self.run:
            return
        ended = datetime.utcnow()
        self.run.ended_at = ended
        self.run.status = status
        if self._started_at:
            self.run.duration_ms = int((ended - self._started_at).total_seconds() * 1000)
        if message:
            self.run.message = (self.run.message or "") + message
        if extra_stats:
            self.update_stats(extra_stats)  # commit内で再度commitされてもOK
        db.session.commit()
