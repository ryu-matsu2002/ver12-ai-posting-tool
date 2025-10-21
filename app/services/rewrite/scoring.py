"""
rewrite/scoring.py
────────────────────────────
AIリライト対象を自動スコアリングして
ArticleRewritePlan にUPSERTする。
────────────────────────────
"""

from datetime import datetime
from sqlalchemy import func, and_
from app import db
from app.models import (
    Article,
    Site,
    GSCMetric,
    GSCUrlStatus,
    ArticleRewritePlan,
)

def build_rewrite_candidates(user_id=None, site_id=None, limit=10000):
    """
    各記事のスコアを算出して article_rewrite_plans にUPSERTする。
    """
    # === 対象記事の基本クエリ ===
    query = (
        db.session.query(
            Article.id.label("article_id"),
            Article.site_id,
            Article.user_id,
            Article.title,
            Article.keyword,
            func.coalesce(func.date_part("day", func.age(func.now(), Article.posted_at)), 0).label("age_days"),
            GSCUrlStatus.indexed.label("indexed"),
            GSCMetric.ctr.label("ctr28"),
            GSCMetric.position.label("pos28"),
            GSCMetric.impressions.label("impr28"),
        )
        .outerjoin(GSCUrlStatus, GSCUrlStatus.article_id == Article.id)
        .outerjoin(
            GSCMetric,
            and_(
                GSCMetric.site_id == Article.site_id,
                GSCMetric.user_id == Article.user_id,
                func.lower(GSCMetric.query) == func.lower(Article.keyword),
            ),
        )
        .filter(Article.status == "posted")
    )

    if user_id:
        query = query.filter(Article.user_id == user_id)
    if site_id:
        query = query.filter(Article.site_id == site_id)

    query = query.limit(limit)
    rows = query.all()

    total_candidates = 0
    for r in rows:
        reasons = []
        score = 0

        # 1. 経過日数
        if r.age_days is not None:
            score += min(r.age_days, 180)
            if r.age_days >= 60:
                reasons.append("age_over_60")

        # 2. インデックス未済
        if r.indexed is False:
            score += 50
            reasons.append("not_indexed")

        # 3. 表示回数ゼロ
        if r.impr28 == 0 or r.impr28 is None:
            score += 30
            reasons.append("no_impressions")

        # 4. CTRが低い
        if r.ctr28 is not None:
            ctr_percent = r.ctr28 * 100
            if ctr_percent < 1.0:
                score += 20
                reasons.append("low_ctr")

        # 5. 平均順位
        if r.pos28 is not None and r.pos28 > 10:
            score += 10
            reasons.append("low_position")

        plan = ArticleRewritePlan.query.filter_by(article_id=r.article_id, is_active=True).first()
        if plan:
            plan.priority_score = score
            plan.reason_codes = reasons
            plan.updated_at = datetime.utcnow()
        else:
            plan = ArticleRewritePlan(
                user_id=r.user_id,
                site_id=r.site_id,
                article_id=r.article_id,
                priority_score=score,
                reason_codes=reasons,
                status="queued",
                is_active=True,
                created_at=datetime.utcnow(),
            )
            db.session.add(plan)

        total_candidates += 1

    db.session.commit()
    return total_candidates
