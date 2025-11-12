"""
# app/services/rewrite/scoring.py
────────────────────────────
AIリライト対象を自動スコアリングして
ArticleRewritePlan にUPSERTする。
────────────────────────────
（改修）スコアリングの“前段”で、WPに実在・公開・非topicに限定
"""

from datetime import datetime
from sqlalchemy import func, and_, exists, not_
from app import db
from app.models import (
    Article,
    Site,
    GSCMetric,
    GSCUrlStatus,
    ArticleRewritePlan,
    ContentIndex,   # ★ 追加：WP実在・公開・topic除外の判定に使用
)

def build_rewrite_candidates(user_id=None, site_id=None, limit=10000):
    """
    各記事のスコアを算出して article_rewrite_plans にUPSERTする。
    """
    # === 対象記事の基本クエリ（WP実在・公開・非topicを前処理で限定） ===
    #
    # 条件：
    #  - ContentIndex.site_id = Article.site_id
    #  - ContentIndex.wp_post_id = Article.wp_post_id（＝WP上に“実在”）
    #  - ContentIndex.status = 'publish'（公開）
    #  - ContentIndex.url NOT ILIKE '%topic%'（topic除外）
    # ※ 内部SEOと同方針。is_topic_url相当はSQL側のILIKEで代替。
    ci_exists = exists().where(
        and_(
            ContentIndex.site_id == Article.site_id,
            ContentIndex.wp_post_id == Article.wp_post_id,
            ContentIndex.status == 'publish',
            not_(ContentIndex.url.ilike('%topic%')),
        )
    )
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
        # ★ 追加：WP実在・公開・非topicに限定（未投稿/ドラフトをここで除外）
        .filter(ci_exists)
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
