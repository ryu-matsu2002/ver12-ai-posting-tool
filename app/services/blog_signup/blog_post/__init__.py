# -*- coding: utf-8 -*-
"""
blog_post パッケージの投稿ディスパッチャ
===============================================================
post_blog_article(blog_type, account, title, body_html, image_path=None)
  └ 各サービス固有の投稿関数を呼び出し、
      {"ok": True,  "url": "...", "posted_at": datetime}
      もしくは {"ok": False, "error": "..."} を返す。
---------------------------------------------------------------
* 今回は Note のみ実装。将来 Ameba / はてな等は elif を追加するだけで対応。
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from app.models import BlogType, ExternalBlogAccount  # 相対 import


# ── ブログ別 poster を import ───────────────────────────────
from .note_post import post_note_article
from .hatena_post import post_hatena_article
# 例）Ameba を追加する場合：
# from .ameba_post import post_ameba_article

__all__ = ["post_blog_article"]


def post_blog_article(
    blog_type: BlogType,
    account: ExternalBlogAccount,
    title: str,
    body_html: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    blog_type   : BlogType  (Enum: NOTE, AMEBA, …)
    account     : ExternalBlogAccount
        cookie_path などログイン済み情報を保持
    title       : str  投稿タイトル
    body_html   : str  本文 (HTML)
    image_path  : str | None アイキャッチ画像ファイルパス

    Returns
    -------
    dict
        成功 -> {"ok": True, "url": "...", "posted_at": datetime}
        失敗 -> {"ok": False, "error": "..."}
    """
    if blog_type == BlogType.NOTE:
        return post_note_article(account, title, body_html, image_path)
    
    elif blog_type == BlogType.HATENA:                  # ★追加
        return post_hatena_article(account, title, body_html, image_path)

    # ここに elif を追加していく
    # elif blog_type == BlogType.AMEBA:
    #     return post_ameba_article(account, title, body_html, image_path)

    return {"ok": False, "error": f"Unsupported blog_type: {blog_type.name}"}
