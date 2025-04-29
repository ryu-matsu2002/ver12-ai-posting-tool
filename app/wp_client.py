# ───────────────────────────────────────────────────────────────
# app/wp_client.py
# WordPress 投稿ユーティリティ (fixed v2)
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
# Basic 認証ヘッダー生成
# ──────────────────────────────
def _basic_auth_header(username: str, app_pass: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

# ──────────────────────────────
# アイキャッチ画像アップロード
# ──────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    # 1) 画像をダウンロード
    try:
        dl_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://pixabay.com/",
        }
        resp = requests.get(image_url, headers=dl_headers, timeout=TIMEOUT)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        current_app.logger.error("◆Featured image download failed: %s", e)
        return None

    # 2) multipart/form-data で WP にアップロード
    filename = os.path.basename(image_url.split("?")[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"
    endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"

    files = {
        "file": (filename, content, ctype)
    }
    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Accept": "application/json",
    }

    try:
        up = requests.post(endpoint, headers=headers, files=files, timeout=TIMEOUT)
        current_app.logger.debug("Media upload status: %s", up.status_code)
        current_app.logger.debug("Media upload response: %s", up.text)
        up.raise_for_status()
        return up.json().get("id")
    except Exception as e:
        current_app.logger.error("◆Media upload failed [%s]: %s", getattr(up, "status_code", None), getattr(up, "text", e))
        return None

def _upload_featured_image_with_retry(site, image_url: str, retries=3, delay=5) -> Optional[int]:
    for attempt in range(1, retries+1):
        fid = _upload_featured_image(site, image_url)
        if fid:
            return fid
        current_app.logger.warning("Attempt %d to upload image failed, retrying...", attempt)
        time.sleep(delay)
    return None

# ──────────────────────────────
# HTML 装飾インジェクション
# ──────────────────────────────
_H2_STYLE = 'style="background:#dbeafe;..."'
_H3_STYLE = 'style="background:#e0e7ff;..."'
_P_CLASS  = 'class="tw-paragraph mb-6 leading-relaxed"'
_tag_pat  = re.compile(r"<(/?)(h2|h3|p)(\s[^>]*)?>", re.I)

def _decorate_html(html: str) -> str:
    def _repl(m: re.Match) -> str:
        close, tag, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if close:
            return f"</{tag}>"
        attrs = attrs.strip()
        if tag=="h2" and "style=" not in attrs:
            attrs = f"{attrs} {_H2_STYLE}".strip()
        if tag=="h3" and "style=" not in attrs:
            attrs = f"{attrs} {_H3_STYLE}".strip()
        if tag=="p" and "class=" not in attrs:
            attrs = f"{attrs} {_P_CLASS}".strip()
        return f"<{tag} {attrs}>".replace("  "," ")
    return _tag_pat.sub(_repl, html)

# ──────────────────────────────
# 記事投稿
# ──────────────────────────────
def post_to_wp(site, article) -> str:
    # 1) アイキャッチ
    featured_id = None
    if article.image_url:
        featured_id = _upload_featured_image_with_retry(site, article.image_url)

    # 2) 本文装飾
    styled = _decorate_html(article.body or "")

    # 3) 投稿ペイロード
    api = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"
    payload = {"title": article.title, "content": styled, "status": "publish"}
    if featured_id:
        payload["featured_media"] = featured_id

    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    current_app.logger.debug("WP POST → %s", api)
    current_app.logger.debug("Headers: %s\nPayload: %s", headers, payload)

    try:
        r = requests.post(api, headers=headers, json=payload, timeout=TIMEOUT)
        current_app.logger.debug("WP resp status: %s body: %s", r.status_code, r.text)
        r.raise_for_status()
        return r.json().get("link","")
    except HTTPError as e:
        current_app.logger.error("WordPress 投稿 failed [%s]: %s", r.status_code, r.text)
        raise

# ──────────────────────────────
# 画像 DL リトライ (未使用)
# ──────────────────────────────
def _download_with_retry(url: str, retries=3, delay=5):
    dl_headers = {"User-Agent":"Mozilla/5.0","Referer":"https://pixabay.com/"}
    for i in range(retries):
        try:
            r = requests.get(url, headers=dl_headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            current_app.logger.warning("Download attempt %d failed: %s", i+1, e)
            time.sleep(delay)
    return None
