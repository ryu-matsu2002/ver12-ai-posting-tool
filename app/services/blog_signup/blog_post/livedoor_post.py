"""
app/services/blog_post/livedoor_post.py
---------------------------------------
ExternalBlogAccount が blog_type == BlogType.LIVEDOOR のとき、
ExternalArticleSchedule を実行して記事を AtomPub API で投稿する。

依存:
    * app.services.livedoor_atompub.post_entry
    * app.services.blog_signup.crypto_utils.decrypt
    * SQLAlchemy セッション (db)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any

from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount, ExternalArticleSchedule, Article
from app.services.blog_signup.crypto_utils import decrypt
from app.services.livedoor_atompub import post_entry

logger = logging.getLogger(__name__)


def post_blog_article(
    blog_account: ExternalBlogAccount,
    schedule: ExternalArticleSchedule,
    article: Article,
) -> Dict[str, Any]:
    """
    1 記事を投稿して Schedule・Article を更新する。
    成功すると dict(result='success', url='...') を返す。
    失敗時は dict(result='error', message='...') を返し、呼び出し元でリトライ判断。
    """

    assert blog_account.blog_type == BlogType.LIVEDOOR, "blog_type mismatch"

    # ── 投稿先情報（復号）
    blog_id: str = blog_account.livedoor_blog_id
    api_key_enc: str = blog_account.atompub_key_enc
    if not (blog_id and api_key_enc):
        msg = "blog_id / api_key が未登録のため投稿できません"
        logger.error(msg)
        return {"result": "error", "message": msg}

    try:
        # ── AtomPub 投稿
        article_id, public_url = post_entry(
            blog_id=blog_id,
            api_key_enc=api_key_enc,
            title=article.title,
            content=article.content,
            categories=[cat.strip() for cat in (article.tags or "").split(",") if cat.strip()],
        )

        # ── DB 更新
        schedule.status = "posted"
        schedule.posted_at = datetime.utcnow()
        schedule.posted_url = public_url
        blog_account.posted_cnt += 1
        db.session.commit()

        logger.info("[LD-Post] site=%s kw=%s → %s",
                    blog_account.site_id, schedule.keyword_id, public_url)

        return {"result": "success", "url": public_url}

    except Exception as e:  # broad on purpose → 呼び元で再判定
        logger.exception("[LD-Post] failed: %s", e)
        schedule.status = "error"
        schedule.message = str(e)
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
        return {"result": "error", "message": str(e)}
