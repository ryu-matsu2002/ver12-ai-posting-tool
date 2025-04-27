# app/wp_client.py

"""
WordPress 投稿ユーティリティ
────────────────────────────────────────
・Basic 認証 (Application Password) 前提
・アイキャッチ → /media → ID を取得
・本文に <h2>,<h3>,<p> のスタイル/クラスをインライン付与
"""

from __future__ import annotations
import base64
import mimetypes
import os
import re
import logging
from typing import Optional

import requests
from requests.exceptions import HTTPError
from requests.auth import HTTPBasicAuth
from flask import current_app

# タイムアウト（秒）
TIMEOUT = int(os.getenv("WP_API_TIMEOUT", "15"))


# ──────────────────────────────
# 認証ヘッダー (アイキャッチ用アップロードでも利用可)
# ──────────────────────────────
def _basic_auth_header(username: str, app_pass: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ──────────────────────────────
# アイキャッチ画像アップロード
# ──────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    try:
        r = requests.get(image_url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        current_app.logger.error("Featured image download failed: %s", e)
        return None

    filename = os.path.basename(image_url.split("?")[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"

    # Basic auth for media upload
    auth = HTTPBasicAuth(site.username, site.app_pass)
    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }
    endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"

    current_app.logger.debug("Media upload → URL: %s", endpoint)
    current_app.logger.debug("Media headers: %s", headers)

    try:
        up = requests.post(endpoint, headers=headers, auth=auth, data=r.content, timeout=TIMEOUT)
        current_app.logger.debug("Media upload response: %s", up.status_code)
        current_app.logger.debug("Media upload body: %s", up.text)
        up.raise_for_status()
        media_id = up.json().get("id")
        current_app.logger.debug("Uploaded media ID: %s", media_id)
        return media_id
    except Exception as e:
        current_app.logger.error("Media upload failed [%s]: %s", getattr(up, "status_code", None), getattr(up, "text", e))
        return None


# ──────────────────────────────
# HTML に装飾用の style / class を注入
# ──────────────────────────────
_H2_STYLE  = (
    'style="background:#dbeafe;color:#1e3a8a;'
    'padding:.75rem 1.25rem;border-left:4px solid #3b82f6;'
    'border-radius:.375rem;font-weight:600;'
    'font-size:1.5rem;margin:1.75rem 0 .75rem"'
)
_H3_STYLE  = (
    'style="background:#e0e7ff;color:#312e81;'
    'padding:.5rem 1rem;border-left:3px solid #6366f1;'
    'border-radius:.375rem;font-weight:500;'
    'font-size:1.25rem;margin:1.5rem 0 .5rem"'
)
_P_CLASS   = 'class="tw-paragraph mb-6 leading-relaxed"'
_tag_pat   = re.compile(r"<(/?)(h2|h3|p)(\s[^>]*)?>", re.I)

def _decorate_html(html: str) -> str:
    """
    <h2>,<h3>,<p> に装飾をインライン付与。
    既に style / class がある場合は追記しない（簡易判定）。
    """
    def _repl(m: re.Match) -> str:
        close, tag, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if close:
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
    site: models.Site インスタンス（url, username, app_pass）
    article: models.Article インスタンス（title, body, image_url）
    戻り値: 投稿先の記事URL
    """
    # 1. アイキャッチ
    featured_id: Optional[int] = None
    if article.image_url:
        featured_id = _upload_featured_image(site, article.image_url)

    # 2. 本文装飾
    styled_body = _decorate_html(article.body or "")

    # 3. 投稿 API
    api = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"
    payload = {
        "title":   article.title,
        "content": styled_body,
        "status":  "publish",
    }
    if featured_id:
        payload["featured_media"] = featured_id

    # Basic 認証
    auth = HTTPBasicAuth(site.username, site.app_pass)
    headers = {
        "Accept":       "application/json",
        "Content-Type": "application/json",
    }

    # デバッグログ: リクエスト詳細
    current_app.logger.debug("WP POST URL: %s", api)
    current_app.logger.debug("WP Payload: %s", payload)
    current_app.logger.debug("WP Username: %s", site.username)

    resp = requests.post(
        api,
        json=payload,
        auth=auth,
        headers=headers,
        timeout=TIMEOUT
    )

    # デバッグログ: レスポンス詳細
    current_app.logger.debug("WP Response status: %s", resp.status_code)
    current_app.logger.debug("WP Response body: %s", resp.text)

    try:
        resp.raise_for_status()
    except HTTPError:
        current_app.logger.error("WordPress 投稿失敗 [%s]: %s", resp.status_code, resp.text)
        raise

    return resp.json().get("link", "")
