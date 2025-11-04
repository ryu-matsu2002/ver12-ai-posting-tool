# app/services/rewrite/bulk_runner.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone

from flask import current_app
from sqlalchemy import or_
from sqlalchemy.orm import selectinload

from app import db
from app.models import Article, ArticleRewritePlan
from app.services.rewrite import serp_collector as serp
from app.services.rewrite import executor as rewrite_executor


def enqueue_user_rewrite(
    user_id: int,
    site_ids: Optional[List[int]] = None,
    article_ids: Optional[List[int]] = None,
    priority_score: float = 0.0,
) -> Dict[str, Any]:
    """
    指定ユーザー配下の対象記事に対し、重複なく ArticleRewritePlan を 'queued' で投入。
    既に queued/running の計画がある記事は自動スキップ。
    """
    with current_app.app_context():
        q = Article.query.filter(Article.user_id == user_id)
        if site_ids:
            q = q.filter(Article.site_id.in_(site_ids))
        if article_ids:
            q = q.filter(Article.id.in_(article_ids))

        # 既存 queued/running がある記事は除外
        sub = (
            db.session.query(ArticleRewritePlan.id)
            .filter(
                ArticleRewritePlan.article_id == Article.id,
                ArticleRewritePlan.is_active.is_(True),
                ArticleRewritePlan.status.in_(["queued", "running"]),
            )
            .exists()
        )
        q = q.filter(~sub)

        targets: List[Article] = q.options(selectinload(Article.site)).all()
        enq = 0
        for art in targets:
            plan = ArticleRewritePlan(
                user_id=user_id,
                site_id=art.site_id,
                article_id=art.id,
                status="queued",
                is_active=True,
                attempts=0,
                priority_score=priority_score,
                created_at=datetime.now(timezone.utc),
            )
            db.session.add(plan)
            enq += 1
        db.session.commit()
        return {"enqueued": enq, "skipped": 0}


def _pick_next_plan_id() -> Optional[Tuple[int, int]]:
    """
    次に実行すべき queued 計画を 1 件返す。(plan_id, user_id)
    """
    row = (
        db.session.query(ArticleRewritePlan.id, ArticleRewritePlan.user_id)
        .filter(
            ArticleRewritePlan.is_active.is_(True),
            ArticleRewritePlan.status == "queued",
        )
        .order_by(ArticleRewritePlan.priority_score.desc(),
                  ArticleRewritePlan.created_at.asc())
        .limit(1)
        .first()
    )
    return (row[0], row[1]) if row else None


def rewrite_tick_once(app, *, dry_run: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """
    キューの次の 1 件を実行。なければ None。
    """
    if dry_run is None:
        dry_run = (os.getenv("REWRITE_DRYRUN", "0") == "1")
    with app.app_context():
        picked = _pick_next_plan_id()
        if not picked:
            return None
        plan_id, user_id = picked
        res = rewrite_executor.execute_one_plan(user_id=user_id, plan_id=plan_id, dry_run=dry_run)
        return {"plan_id": plan_id, "user_id": user_id, "result": res}


def serp_warmup_for_recent_articles(app, *, days: int = 45, limit_per_run: int = 30) -> Dict[str, Any]:
    """
    直近 N 日に更新/投稿された記事のうち上限件数を選び、SERPアウトラインを再収集してキャッシュを温める。
    """
    with app.app_context():
        since = datetime.now(timezone.utc) - timedelta(days=days)
        ids = [
            r[0] for r in (
                db.session.query(Article.id)
                .filter(or_(Article.updated_at >= since, Article.posted_at >= since))
                .order_by(Article.updated_at.desc().nullslast())
                .limit(limit_per_run)
                .all()
            )
        ]
        ok = 0
        for aid in ids:
            try:
                r = serp.collect_and_cache_for_article(aid, force=True, limit=6, lang="ja", gl="jp")
                if r.get("ok"):
                    ok += 1
            except Exception as e:
                current_app.logger.info("[rewrite/serp_warmup] aid=%s error=%r", aid, e)
        return {"touched": len(ids), "warmed": ok}


def retry_failed_plans(
    app,
    *,
    max_attempts: int = 3,
    min_age_minutes: int = 30,
    to_queue_limit: int = 50,
) -> Dict[str, Any]:
    """
    失敗（status='error'）かつ一定時間経過・試行回数上限未満の計画を queued に戻す。
    """
    with app.app_context():
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
        # updated_at はテーブルに無いので、終了時刻/開始時刻/作成時刻のいずれかが十分古いものを対象にする
        rows = (
            db.session.query(ArticleRewritePlan)
            .filter(
                ArticleRewritePlan.status == "error",
                or_(ArticleRewritePlan.attempts.is_(None), ArticleRewritePlan.attempts < max_attempts),
                or_(
                    ArticleRewritePlan.finished_at <= cutoff,
                    ArticleRewritePlan.started_at  <= cutoff,
                    ArticleRewritePlan.created_at  <= cutoff,
                ),
            )
            .order_by(
                ArticleRewritePlan.finished_at.asc().nullslast(),
                ArticleRewritePlan.started_at.asc().nullslast(),
                ArticleRewritePlan.created_at.asc(),
            )
            .limit(to_queue_limit)
            .all()
        )
        for p in rows:
            p.status = "queued"
            p.started_at = None
            p.finished_at = None
            # attempts は実行側でインクリメントされる想定のまま維持
        db.session.commit()
        return {"requeued": len(rows)}
