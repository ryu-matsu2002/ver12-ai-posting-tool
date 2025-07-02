# app/services/blog_post/__init__.py
"""
ブログ種別ごとの投稿ディスパッチャ
post_blog_article(...) を呼ぶだけで、
BlogType に応じた poster が実行される。
"""

import logging
from enum import Enum
from typing import Dict, Any

from app.models import BlogType         # Enum(NOTE / HATENA / …)

# ---- ブログ別 poster を import ----
from .note_poster import post_to_note   # いまは Note のみ実装

# --------------------------------------------------------------
def post_blog_article(
    blog_type: BlogType,
    title: str,
    body_html: str,
    email: str,
    password: str
) -> Dict[str, Any]:
    """
    Args:
        blog_type : BlogType Enum
        title     : 記事タイトル
        body_html : HTML 本文
        email     : ブログアカウントのメール
        password  : ブログアカウントのパスワード

    Returns:
        poster が返す dict
        {"ok": True/False, "url": ..., "error": ...}
    """
    if blog_type == BlogType.NOTE:
        return post_to_note(title, body_html, email, password)

    # まだ未実装のブログタイプ
    msg = f"[post_blog_article] blog_type {blog_type} not supported"
    logging.warning(msg)
    return {"ok": False, "error": msg}


__all__ = ["post_blog_article"]
