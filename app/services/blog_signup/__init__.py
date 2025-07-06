"""
blog_signup パッケージ公開 API
------------------------------------------------
register_blog_account(site_id, blog_type)
  └ ブログ種別ごとにアカウント登録を振り分ける同期同期関数
"""

from __future__ import annotations

import asyncio
import secrets
import string

from app import db
from app.models import BlogType, ExternalBlogAccount
from .note_signup import signup_note_account  # async coroutine

# ──────────────────────────────────────────
def _random_email() -> str:
    # disposable ドメインだと弾かれるため汎用ドメインを使用
    prefix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    return f"{prefix}@example.net"


def _random_password() -> str:
    # 8 文字以上・英数＋記号を必ず含む
    core = secrets.token_urlsafe(10)[:8]
    return core + "A1!"


# ──────────────────────────────────────────
def register_note_account_sync(site_id: int) -> dict:
    """
    Note アカウントを同期で登録し、結果 dict を返す
    """
    # 1) DB にレコードを作成して ID を確定
    email    = _random_email()
    password = _random_password()

    acct = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.NOTE,
        email     = email,
        password  = password,
        username  = email.split("@")[0],
        nickname  = email.split("@")[0],
        status    = "pending",
    )
    db.session.add(acct)
    db.session.commit()

    # 2) Playwright でサインアップ実行
    result = asyncio.run(signup_note_account(acct))

    # 3) 成否に応じて処理
    if result.get("ok"):
        return {
            "ok": True,
            "account_id": acct.id,
            "email": email,
            "password": password,
        }

    # 失敗時はレコードを無効化
    acct.status = "error"
    db.session.commit()
    return {"ok": False, "error": result.get("error", "unknown")}


# ──────────────────────────────────────────
def register_blog_account(site_id: int, blog_type: BlogType):
    """ブログ種別ディスパッチャ"""
    if blog_type == BlogType.NOTE:
        return register_note_account_sync(site_id)

    raise ValueError(f"Blog type {blog_type} not supported yet.")


__all__ = ["register_blog_account"]
