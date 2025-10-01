# app/services/internal_seo/user_scheduler.py
"""
User-scope Internal SEO scheduler (tick loop)

ユーザー単位の内部SEOスケジュールを実行する本体実装。
- 毎分などでスキャンして due なユーザーを tick 実行（APScheduler から呼ばれる想定）
- ユーザー毎の排他を Redis Lock で担保
- 進捗は InternalSeoUserRun / InternalSeoUserSchedule に記録
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
from flask import current_app
from typing import Optional, Dict, Any


TICK_JOB_ID = "internal_seo_user_tick"
DEFAULT_TICK_SEC = int(os.getenv("INTERNAL_SEO_USER_TICK_SEC", "60"))
DEFAULT_BUDGET   = int(os.getenv("INTERNAL_SEO_USER_BUDGET", "20"))
ERROR_BACKOFF    = int(os.getenv("INTERNAL_SEO_USER_ERROR_BACKOFF_SEC", "300"))
# pending=0 のときに次回 tick を先送りする待機秒数（空振り防止）
# 例: 15分（900秒）
DEFAULT_IDLE_BACKOFF = int(os.getenv("INTERNAL_SEO_USER_IDLE_BACKOFF_SEC", "900"))


# ------------------------------------------------------------
# 補助関数
# ------------------------------------------------------------

def _count_user_pending_posts(user_id: int) -> int:
    """ユーザー配下サイトの pending (distinct post_id) 件数を返す。"""
    from app import db
    from sqlalchemy import text
    sql = text(
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT a.post_id
            FROM internal_link_actions a
            JOIN site s ON s.id = a.site_id
           WHERE s.user_id = :uid
             AND a.status = 'pending'
        ) t
        """
    )
    return int(db.session.execute(sql, {"uid": user_id}).scalar() or 0)


def _with_rate_limit_override(rate_limit_per_min: Optional[int]):
    """
    スケジュール個別のレート上限を一時的に環境変数へ反映するコンテキスト。
    """
    class _Ctx:
        def __enter__(self):
            self._prev = os.getenv("INTERNAL_SEO_RATE_LIMIT_PER_MIN", None)
            if rate_limit_per_min is not None:
                os.environ["INTERNAL_SEO_RATE_LIMIT_PER_MIN"] = str(int(rate_limit_per_min))
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._prev is None:
                os.environ.pop("INTERNAL_SEO_RATE_LIMIT_PER_MIN", None)
            else:
                os.environ["INTERNAL_SEO_RATE_LIMIT_PER_MIN"] = self._prev
    return _Ctx()


# ------------------------------------------------------------
# ユーザー単位 tick 実行
# ------------------------------------------------------------

def run_user_tick(app, user_id: int, *, force: bool = False) -> Dict[str, Any]:
    """
    単一ユーザーの 1 tick を実行する。
    - Redis Lock により多重起動を防止
    - apply_actions_for_user を実行し、Run/Schedule を更新
    - 戻り値: メトリクス辞書（UI用の軽いサマリ）
    """
    from app import db
    from app.models import InternalSeoUserSchedule, InternalSeoUserRun
    from app.services.internal_seo.applier import apply_actions_for_user
    from app.utils.redis_lock import redis_lock
    from sqlalchemy.exc import IntegrityError

    now_utc = datetime.now(timezone.utc)

    with app.app_context():
        sched: Optional[InternalSeoUserSchedule] = (
            InternalSeoUserSchedule.query.filter_by(user_id=user_id).one_or_none()
        )
        if not sched:
            try:
                sched = InternalSeoUserSchedule(
                    user_id=user_id,
                    is_enabled=False,
                    status="idle",
                    tick_interval_sec=int(os.getenv("INTERNAL_SEO_USER_TICK_SEC", "60")),
                    budget_per_tick=int(os.getenv("INTERNAL_SEO_USER_BUDGET", "20")),
                )
                db.session.add(sched)
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                sched = InternalSeoUserSchedule.query.filter_by(user_id=user_id).one_or_none()
        if not sched:
            current_app.logger.warning("[iseo-user] schedule missing and cannot create (user=%s)", user_id)
            return {"ok": False, "reason": "schedule-missing"}
        
        # 参照系の軽いメトリクス（ログ用）：tick前の pending 記事数
        remain_before = _count_user_pending_posts(user_id)
        current_app.logger.info(
            "[iseo-user] tick start user=%s force=%s sched(is_enabled=%s, status=%s, next=%s) remain_before=%s",
            user_id, bool(force), bool(sched.is_enabled), sched.status, 
            sched.next_run_at.isoformat() if sched.next_run_at else None,
            remain_before,
        )

        if not force:
            if not sched.is_enabled:
                current_app.logger.info("[iseo-user] skip user=%s reason=disabled remain=%s", user_id, remain_before)
                return {"ok": False, "reason": "disabled"}
            if sched.status == "paused":
                current_app.logger.info("[iseo-user] skip user=%s reason=paused remain=%s", user_id, remain_before)
                return {"ok": False, "reason": "paused"}
            if sched.next_run_at and sched.next_run_at > now_utc:
                current_app.logger.info(
                    "[iseo-user] skip user=%s reason=not-due next=%s remain=%s",
                    user_id, sched.next_run_at.isoformat(), remain_before
                )
                return {"ok": False, "reason": "not-due"}
            
        # ── 追加: 無弾（pending=0）の早期リターン（効率化・挙動不変）
        # ここではスケジュールを idle にし、アイドル待機(backoff)だけを更新する。
        # ※ is_enabled=False や paused の場合は上で既に return 済み
        remain_peek = remain_before
        if remain_peek == 0 and not force:
            # 次回スケジュールだけ決めて即終了（ロック/適用を走らせない）
            interval_sec = int(sched.tick_interval_sec or DEFAULT_TICK_SEC)
            idle_backoff = DEFAULT_IDLE_BACKOFF
            now_utc = datetime.now(timezone.utc)
            sched.status = "idle"
            sched.next_run_at = now_utc + timedelta(seconds=idle_backoff)
            sched.last_run_at = now_utc
            sched.last_error = None
            db.session.commit()
            current_app.logger.info(
                "[iseo-user] early-idle user=%s (no pending) next=%s",
                user_id,
                (sched.next_run_at.isoformat() if sched.next_run_at else None),
            )
            return {
                "ok": True,
                "applied": 0,
                "swapped": 0,
                "skipped": 0,
                "processed_posts": 0,
                "remain_posts": 0,
                "next_run_at": sched.next_run_at.isoformat() if sched.next_run_at else None,
            }

        lock_key = f"iseo:user:{user_id}:lock"
        lock_ttl = int(os.getenv("INTERNAL_SEO_USER_LOCK_TTL_SEC", "900"))  # 15分

        with redis_lock(lock_key, ttl=lock_ttl, wait=True, wait_timeout=5.0) as acquired:
            if not acquired:
                current_app.logger.info("[iseo-user] skip user=%s reason=locked remain=%s", user_id, remain_before)
                return {"ok": False, "reason": "locked"}

            sched.status = "running"
            sched.last_error = None
            sched.last_run_at = now_utc
            db.session.commit()

            run = InternalSeoUserRun(
                user_id=user_id,
                started_at=now_utc,
                status="running",
            )
            db.session.add(run)
            db.session.commit()

            applied = swapped = skipped = processed = 0
            notes: Dict[str, Any] = {}

            try:
                budget = int(sched.budget_per_tick or DEFAULT_BUDGET)
                rlim   = sched.rate_limit_per_min  # None の可能性あり
                current_app.logger.info(
                    "[iseo-user] run user=%s budget=%s rate_limit_per_min=%s",
                    user_id, budget, rlim if rlim is not None else "(default)"
                )
                with _with_rate_limit_override(rlim):
                    result = apply_actions_for_user(user_id=user_id, limit_posts=budget, dry_run=False)

                applied   = int(result.get("applied", 0) or 0)
                swapped   = int(result.get("swapped", 0) or 0)
                skipped   = int(result.get("skipped", 0) or 0)
                processed = int(result.get("processed_posts", 0) or 0)
                for k in ("site_breakdown", "stats"):
                    if k in result:
                        notes[k] = result[k]

                run.applied = applied
                run.swapped = swapped
                run.skipped = skipped
                run.processed_posts = processed
                run.finished_at = datetime.now(timezone.utc)

                # 次回スケジュール決定（pending=0 でも先送りして空振りループを防止）
                remain       = _count_user_pending_posts(user_id)
                interval_sec = int(sched.tick_interval_sec or DEFAULT_TICK_SEC)
                idle_backoff = DEFAULT_IDLE_BACKOFF
                if sched.is_enabled:
                    if remain > 0:
                        # まだ弾がある＝通常の tick 間隔で回す
                        sched.status     = "running"
                        sched.next_run_at = now_utc + timedelta(seconds=interval_sec)
                    else:
                        # 弾が枯渇している＝アイドル待機（やや長め）で再チェック
                        sched.status     = "idle"
                        sched.next_run_at = now_utc + timedelta(seconds=idle_backoff)
                else:
                    # 明示的に無効化中は次回未設定
                    sched.status     = "idle"
                    sched.next_run_at = None

                run.status = "success"
                run.notes = notes or {}
                db.session.commit()
                current_app.logger.info(
                    "[iseo-user] done user=%s applied=%s swapped=%s skipped=%s processed_posts=%s "
                    "remain_after=%s next=%s",
                    user_id, applied, swapped, skipped, processed, remain,
                    (sched.next_run_at.isoformat() if sched.next_run_at else None)
                )

                return {
                    "ok": True,
                    "applied": applied,
                    "swapped": swapped,
                    "skipped": skipped,
                    "processed_posts": processed,
                    "remain_posts": remain,
                    "next_run_at": sched.next_run_at.isoformat() if sched.next_run_at else None,
                }

            except Exception as e:
                db.session.rollback()
                current_app.logger.exception("[iseo-user] tick failed (user=%s): %s", user_id, e)
                run.status = "failed"
                run.finished_at = datetime.now(timezone.utc)
                backoff = ERROR_BACKOFF
                sched.status = "error"
                sched.last_error = str(e)[:1000]
                sched.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
                db.session.commit()
                return {"ok": False, "reason": "error", "message": str(e)}


# ------------------------------------------------------------
# APScheduler エントリポイント
# ------------------------------------------------------------

def user_scheduler_tick(app) -> None:
    """
    APScheduler から毎分呼ばれる想定の「スキャン＆実行」入口。
    """
    from app import db
    from app.models import InternalSeoUserSchedule
    from sqlalchemy import or_

    with app.app_context():
        now_utc = datetime.now(timezone.utc)
        current_app.logger.info(
            "[iseo-user-scheduler] tick fired at %s (job_id=%s)",
            now_utc.isoformat(),
            TICK_JOB_ID,
        )
        q = (
            InternalSeoUserSchedule.query
            .filter(InternalSeoUserSchedule.is_enabled.is_(True))
            .filter(or_(InternalSeoUserSchedule.next_run_at.is_(None),
                        InternalSeoUserSchedule.next_run_at <= now_utc))
            .order_by(InternalSeoUserSchedule.next_run_at.asc().nullsfirst())
        )
        scan_limit = int(os.getenv("INTERNAL_SEO_USER_SCAN_LIMIT", "50"))
        rows = q.limit(scan_limit).all()
        current_app.logger.info("[iseo-user-scheduler] due_users=%s scan_limit=%s", len(rows), scan_limit)
        if not rows:
            current_app.logger.info("[iseo-user-scheduler] no due users")
            return

        for sched in rows:
            try:
                res = run_user_tick(app, sched.user_id)
                # 重要指標のみを抜粋してログを簡潔に
                if res.get("ok"):
                    current_app.logger.info(
                        "[iseo-user-scheduler] user=%s ok applied=%s swapped=%s skipped=%s processed=%s remain=%s next=%s",
                        sched.user_id,
                        res.get("applied"), res.get("swapped"), res.get("skipped"),
                        res.get("processed_posts"), res.get("remain_posts"),
                        res.get("next_run_at"),
                    )
                else:
                    current_app.logger.info(
                        "[iseo-user-scheduler] user=%s skip_or_error reason=%s",
                        sched.user_id, res.get("reason")
                    )
            except Exception as e:
                current_app.logger.exception("[iseo-user-scheduler] run_user_tick error (user=%s): %s", sched.user_id, e)
        db.session.close()
