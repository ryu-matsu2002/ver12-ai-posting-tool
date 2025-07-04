"""
blog_signup パッケージ公開 API
---------------------------------
register_blog_account(site_id, blog_type)
    ブログ種別ごとにアカウント登録を振り分ける同期関数
"""

import asyncio
from app import db
from app.models import BlogType
from .note_signup import signup_note_account   # async coroutine

# ──────────────────────────────────────────
def register_note_account_sync(site_id: int) -> dict:
    """
    async 関数 signup_note_account() を同期ラッパで実行。
    site_id をメモする場合はここで DB へ保存するなど拡張可。
    """
    result = asyncio.run(signup_note_account())
    if result.get("ok"):
        from app.models import ExternalBlogAccount
        acct = ExternalBlogAccount(
            site_id   = site_id,
            blog_type = BlogType.NOTE,
            email     = result["email"],
            password  = result["password"],
            username  = result["email"].split("@")[0],
            nickname  = result["email"].split("@")[0],
        )
        db.session.add(acct)
        db.session.commit()
        result["account_id"] = acct.id
    return result

# ──────────────────────────────────────────
def register_blog_account(site_id: int, blog_type: BlogType):
    """共通入口：ブログ種別で振り分け"""
    if blog_type == BlogType.NOTE:
        return register_note_account_sync(site_id)
    raise ValueError(f"Blog type {blog_type} not supported yet.")

__all__ = ["register_blog_account"]
