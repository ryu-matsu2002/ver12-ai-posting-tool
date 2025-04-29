# ───────────────────────────────────────────────────────────────
# app/wp_client.py
# WordPress 投稿ユーティリティ
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import base64
import mimetypes
import os
import re
import logging
import time
from typing import Optional
import requests
from requests.exceptions import HTTPError
from flask import current_app

# タイムアウト（秒）
TIMEOUT = int(os.getenv("WP_API_TIMEOUT", "15"))

# ──────────────────────────────
# Basic 認証ヘッダー自前生成
# ──────────────────────────────
def _basic_auth_header(username: str, app_pass: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

# ──────────────────────────────
# アイキャッチ画像アップロード
# ──────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    try:
        dl_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Referer": "https://pixabay.com/",
        }
        r = requests.get(image_url, headers=dl_headers, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        current_app.logger.error("Featured image download failed: %s", e)
        return None

    filename = os.path.basename(image_url.split("?")[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"
    endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"

    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
        "Accept": "application/json",
    }

    current_app.logger.debug("Media upload → URL: %s", endpoint)
    current_app.logger.debug("Media upload Headers: %s", headers)

    try:
        up = requests.post(endpoint, headers=headers, data=r.content, timeout=TIMEOUT)
        current_app.logger.debug("Media upload Response status: %s", up.status_code)
        current_app.logger.debug("Media upload Response body: %s", up.text)
        up.raise_for_status()
        return up.json().get("id")
    except Exception as e:
        current_app.logger.error(
            "Media upload failed [%s]: %s",
            getattr(up, "status_code", None),
            getattr(up, "text", e)
        )
        return None

# 画像アップロード時のリトライ機能
def _upload_featured_image_with_retry(site, image_url: str, retries=3, delay=5) -> Optional[int]:
    for attempt in range(retries):
        featured_id = _upload_featured_image(site, image_url)
        if featured_id:
            return featured_id
        current_app.logger.error(f"Attempt {attempt + 1} failed to upload image: {image_url}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None  # リトライしても失敗した場合

# ──────────────────────────────
# HTML 装飾インジェクション
# ──────────────────────────────
_H2_STYLE = (
    'style="background:#dbeafe;color:#1e3a8a;'
    'padding:.75rem 1.25rem;border-left:4px solid #3b82f6;'
    'border-radius:.375rem;font-weight:600;'
    'font-size:1.5rem;margin:1.75rem 0 .75rem"'
)
_H3_STYLE = (
    'style="background:#e0e7ff;color:#312e81;'
    'padding:.5rem 1rem;border-left:3px solid #6366f1;'
    'border-radius:.375rem;font-weight:500;'
    'font-size:1.25rem;margin:1.5rem 0 .5rem"'
)
_P_CLASS = 'class="tw-paragraph mb-6 leading-relaxed"'
_tag_pat = re.compile(r"<(/?)(h2|h3|p)(\s[^>]*)?>", re.I)

def _decorate_html(html: str) -> str:
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
    site: models.Site(url, username, app_pass)
    article: models.Article(title, body, image_url)
    戻り値: 投稿先の記事URL
    """

    # 1) アイキャッチ
    featured_id: Optional[int] = None
    if article.image_url:
        featured_id = _upload_featured_image_with_retry(site, article.image_url)

    # 2) 本文装飾
    styled_body = _decorate_html(article.body or "")

    # 3) WordPress REST API 投稿
    api = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"
    payload = {
        "title": article.title,
        "content": styled_body,
        "status": "publish",
    }
    if featured_id:
        payload["featured_media"] = featured_id

    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
                      "Gecko/20100101 Firefox/124.0",
    }

    current_app.logger.debug("WP POST URL: %s", api)
    current_app.logger.debug("WP Request Headers: %s", headers)
    current_app.logger.debug("WP Request Payload: %s", payload)

    try:
        resp = requests.post(api, headers=headers, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        current_app.logger.debug("WP Response status: %s", resp.status_code)
        current_app.logger.debug("WP Response body: %s", resp.text)
    except HTTPError:
        current_app.logger.error("WordPress 投稿失敗 [%s]: %s", resp.status_code, resp.text)
        raise

    return resp.json().get("link", "")

# ──────────────────────────────────────────────
# 画像ダウンロード時のリトライ機能
# ──────────────────────────────────────────────
def _download_with_retry(url: str, retries=3, delay=5):
    dl_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Referer": "https://pixabay.com/",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=dl_headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            current_app.logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
