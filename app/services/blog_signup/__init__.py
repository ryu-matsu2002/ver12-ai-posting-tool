"""
blog_signup パッケージの公開インターフェース
- register_blog_account    … ブログ種別を判断して登録を呼び出すラッパー
- register_note_account    … Note 専用の実体（note_signup.py 内の signup_note_account を別名で公開）
"""

from app.models import BlogType
from .note_signup import signup_note_account as register_note_account  # ✅ エイリアス

def register_blog_account(site_id: int, blog_type: BlogType):
    """
    ブログ種別に応じてアカウント登録処理を振り分ける共通入口。
    ここを tasks.py などが呼び出す。
    """
    if blog_type == BlogType.NOTE:
        return register_note_account(site_id)
    raise ValueError(f"Blog type {blog_type} not supported yet.")

__all__ = [
     "register_blog_account",
     "register_note_account",
]