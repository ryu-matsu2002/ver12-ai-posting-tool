"""
blog_signup パッケージ公開 API
---------------------------------
register_blog_account(site_id, blog_type)
    ブログ種別ごとにアカウント登録を振り分ける同期関数
"""

import asyncio, secrets, string
from app import db
from app.models import BlogType, ExternalBlogAccount
from .note_signup import signup_note_account   # async coroutine

# ──────────────────────────────────────────
def _random_email() -> str:
    prefix = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    return f"{prefix}@example.com"

def _random_password() -> str:
    return secrets.token_urlsafe(12)

# ──────────────────────────────────────────
def register_note_account_sync(site_id: int) -> dict:
    """
    async 関数 signup_note_account(account) を同期ラッパで実行。
    """
    # ① 先にアカウントレコードを作成
    email    = _random_email()
    password = _random_password()

    acct = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.NOTE,
        email     = email,
        password  = password,
        username  = email.split("@")[0],
        nickname  = email.split("@")[0],
    )
    db.session.add(acct)
    db.session.commit()

    # ② Playwright で Note サインアップ（cookie 保存など）
    result = asyncio.run(signup_note_account(acct))   # ← acct を引数に渡す

    # ③ 成功なら OK フラグと account_id を返却
    if result.get("ok"):
        result.update({
            "account_id": acct.id,
            "email": email,
            "password": password
        })
    else:
        # 失敗したらアカウントをロールバック削除しておく
        db.session.delete(acct)
        db.session.commit()

    return result

# ──────────────────────────────────────────
def register_blog_account(site_id: int, blog_type: BlogType):
    """共通入口：ブログ種別で振り分け"""
    if blog_type == BlogType.NOTE:
        return register_note_account_sync(site_id)
    raise ValueError(f"Blog type {blog_type} not supported yet.")

__all__ = ["register_blog_account"]
