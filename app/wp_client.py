# app/wp_client.py
"""
WordPress 投稿ユーティリティ
────────────────────────────────────────
・ユーザが登録した Site モデル (url / username / app_pass) を用いて
  WordPress REST API へ記事を即時投稿するヘルパ関数群
・アプリパスワード (Basic 認証) 前提
・アイキャッチ (image_url) があれば /media へアップロード
・投稿前に <h2>, <h3>, <p> へ装飾用のクラス／インライン style を付与
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from typing import Optional

import requests

# タイムアウト（秒）
TIMEOUT = int(os.getenv("WP_API_TIMEOUT", "15"))

# ──────────────────────────────
# 認証ヘッダー
# ──────────────────────────────
def _basic_auth_header(username: str, app_pass: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ──────────────────────────────
# アイキャッチ画像アップロード
# ──────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    """
    image_url → ダウンロード → site へ /media POST
    成功時 media ID、失敗時 None
    """
    try:
        r = requests.get(image_url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception:
        return None

    filename = os.path.basename(image_url.split("?")[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"

    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }
    endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"

    try:
        up = requests.post(endpoint, headers=headers, data=r.content, timeout=TIMEOUT)
        up.raise_for_status()
        return up.json().get("id")
    except Exception:
        return None


# ──────────────────────────────
# HTML に装飾用の style / class を注入
# ──────────────────────────────
_H2_STYLE  = 'style="background:#dbeafe;color:#1e3a8a;padding:.75rem 1.25rem;border-left:4px solid #3b82f6;border-radius:.375rem;font-weight:600;font-size:1.5rem;margin:1.75rem 0 .75rem"'
_H3_STYLE  = 'style="background:#e0e7ff;color:#312e81;padding:.5rem 1rem;border-left:3px solid #6366f1;border-radius:.375rem;font-weight:500;font-size:1.25rem;margin:1.5rem 0 .5rem"'
_P_CLASS   = 'class="tw-paragraph mb-6 leading-relaxed"'  # Tailwind を利用する場合

_tag_pat   = re.compile(r"<(/?)(h2|h3|p)(\s[^>]*)?>", re.I)


def _decorate_html(html: str) -> str:
    """
    <h2>,<h3>,<p> に装飾をインライン付与。
    既に style / class がある場合は追記しない（簡易判定）。
    """
    def _repl(m: re.Match) -> str:
        close, tag, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if close:  # 閉じタグ
            return f"</{tag}>"

        attrs = attrs.strip()

        if tag == "h2" and "style=" not in attrs:
            attrs = f"{attrs} {_H2_STYLE}".strip()
        elif tag == "h3" and "style=" not in attrs:
            attrs = f"{attrs} {_H3_STYLE}".strip()
        elif tag == "p" and "class=" not in attrs:
            attrs = f'{attrs} {_P_CLASS}'.strip()

        return f"<{tag} {attrs}>".replace("  ", " ")

    return _tag_pat.sub(_repl, html)


# ──────────────────────────────
# 公開 API : 記事投稿
# ──────────────────────────────
def post_to_wp(site, article) -> str:
    """
    Site と Article を受け取り、WordPress へ publish 投稿。
    戻り値: 投稿先 URL（エラー時は例外送出）
    """
    # 1. アイキャッチ
    featured_id: Optional[int] = None
    if article.image_url:
        featured_id = _upload_featured_image(site, article.image_url)

    # 2. コンテンツ装飾
    styled_body = _decorate_html(article.body or "")

    # 3. REST  呼び出し
    api  = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"
    hdrs = _basic_auth_header(site.username, site.app_pass)

    payload: dict = {
        "title":   article.title,
        "content": styled_body,
        "status":  "publish",
    }
    if featured_id:
        payload["featured_media"] = featured_id

    resp = requests.post(api, headers=hdrs, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("link", "")
