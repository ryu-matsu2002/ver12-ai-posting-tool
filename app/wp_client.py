# ───────────────────────────────────────────────────────────────
# app/wp_client.py  (JWT 認証対応版)
# WordPress 投稿ユーティリティ
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import mimetypes
import os
import re
import logging
import time
from typing import Optional, Dict
import requests
from requests.exceptions import HTTPError
from flask import current_app

# タイムアウト（秒）
TIMEOUT = int(os.getenv("WP_API_TIMEOUT", "15"))
# JWT キャッシュ
_jwt_cache: Dict[str, Dict[str, float]] = {}

# ──────────────────────────────────────────────────
# JWT トークン取得・キャッシュ
# ──────────────────────────────────────────────────
def _get_jwt_token(site) -> str:
    """
    JWT 認証トークンを取得し、expires 時刻までキャッシュする
    """
    cache = _jwt_cache.get(site.url)
    now = time.time()
    if cache and cache.get('exp', 0) > now + 10:
        return cache['token']

    endpoint = f"{site.url.rstrip('/')}/wp-json/jwt-auth/v1/token"
    payload = {"username": site.username, "password": site.app_pass}
    resp = requests.post(endpoint, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    token = data.get('token')
    # プラグインが返す expires は Unix タイムスタンプ
    exp = data.get('data', {}).get('expires', now + 3600)
    _jwt_cache[site.url] = {'token': token, 'exp': exp}
    return token

# ──────────────────────────────────────────────────
# アイキャッチ画像アップロード（multipart/form-data）
# ──────────────────────────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    # 1) 画像ダウンロード
    try:
        dl_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://pixabay.com/"}
        resp = requests.get(image_url, headers=dl_headers, timeout=TIMEOUT)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        current_app.logger.error("◆Download failed: %s", e)
        return None

    # 2) WP へ multipart upload
    filename = os.path.basename(image_url.split("?", 1)[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"
    endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"

    files = {"file": (filename, content, ctype)}
    token = _get_jwt_token(site)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    current_app.logger.debug("Media POST to %s files=%s", endpoint, files.keys())
    try:
        up = requests.post(endpoint, headers=headers, files=files, timeout=TIMEOUT)
        current_app.logger.debug("Media upload status: %s body: %s", up.status_code, up.text)
        up.raise_for_status()
        return up.json().get("id")
    except Exception as e:
        current_app.logger.error("◆Media upload failed [%s]: %s", getattr(up, "status_code", None), getattr(up, "text", e))
        return None

# ──────────────────────────────────────────────────
# HTML 装飾
# ──────────────────────────────────────────────────
_H2_STYLE = 'style="background:#dbeafe;..."'
_H3_STYLE = 'style="background:#e0e7ff;..."'
_P_CLASS = 'class="tw-paragraph mb-6 leading-relaxed"'
_tag_pat = re.compile(r"<(/?)(h2|h3|p)(\s[^>]*)?>", re.I)

def _decorate_html(html: str) -> str:
    def _repl(m):
        close, tag, attrs = m.group(1), m.group(2).lower(), m.group(3) or ""
        if close:
            return f"</{tag}>"
        attrs = attrs.strip()
        if tag == "h2" and "style=" not in attrs:
            attrs = f"{attrs} {_H2_STYLE}".strip()
        if tag == "h3" and "style=" not in attrs:
            attrs = f"{attrs} {_H3_STYLE}".strip()
        if tag == "p" and "class=" not in attrs:
            attrs = f"{attrs} {_P_CLASS}".strip()
        return f"<{tag} {attrs}>"
    return _tag_pat.sub(_repl, html)

# ──────────────────────────────────────────────────
# 記事投稿
# ──────────────────────────────────────────────────
def post_to_wp(site, article) -> str:
    # 1) アイキャッチ
    featured_id = None
    if article.image_url:
        featured_id = _upload_featured_image(site, article.image_url)

    # 2) 本文装飾
    styled = _decorate_html(article.body or "")

    # 3) 投稿 API
    api = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"
    payload = {"title": article.title, "content": styled, "status": "publish"}
    if featured_id:
        payload["featured_media"] = featured_id

    token = _get_jwt_token(site)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    current_app.logger.debug("WP POST %s", api)
    current_app.logger.debug("Headers: %s", headers)
    current_app.logger.debug("Payload: %s", payload)

    try:
        r = requests.post(api, headers=headers, json=payload, timeout=TIMEOUT)
        current_app.logger.debug("WP resp %s: %s", r.status_code, r.text)
        r.raise_for_status()
        return r.json().get("link", "")
    except HTTPError:
        current_app.logger.error("WP post failed [%s]: %s", r.status_code, r.text)
        raise

# ──────────────────────────────────────────────────
# 画像 DL retry (unused)
# ──────────────────────────────────────────────────
def _download_with_retry(url: str, retries=3, delay=5):
    dl_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://pixabay.com/"}
    for i in range(retries):
        try:
            rr = requests.get(url, headers=dl_headers, timeout=TIMEOUT)
            rr.raise_for_status()
            return rr
        except Exception as e:
            current_app.logger.warning("Download retry %d failed: %s", i+1, e)
            time.sleep(delay)
    return None
