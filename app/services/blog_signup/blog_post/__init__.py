# app/services/blog_signup/blog_post/__init__.py
"""
ブログ種別ごとの投稿ディスパッチャ
post_blog_article(...) を呼ぶだけで、
BlogType に応じた poster が実行される。
"""

import logging
from typing import Dict, Any
from app.models import BlogType            # Enum(NOTE / HATENA / …)

# ---- ブログ別 poster を import ----
from .note_post import post_to_note        # ← 修正済み

# --------------------------------------------------------------
def post_blog_article(
    blog_type: BlogType,
    title: str,
    body_html: str,
    email: str,
    password: str
) -> Dict[str, Any]:
    """
    Returns:
        {"ok": True/False, "url": ..., "error": ...}
    """
    if blog_type == BlogType.NOTE:
        return post_to_note(title, body_html, email, password)

    msg = f"[post_blog_article] blog_type {blog_type} not supported"
    logging.warning(msg)
    return {"ok": False, "error": msg}


__all__ = ["post_blog_article"]
