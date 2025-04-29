"""
WordPress 投稿ユーティリティ
────────────────────────────────────────
・各ユーザーが登録した Site モデル (url / username / app_pass) を使って
  WordPress REST API へ記事を即時投稿するヘルパ関数群
・アプリパスワード(Basic 認証) を前提
・アイキャッチ( image_url ) があれば /media へ先にアップロード
"""

from __future__ import annotations
import base64
import os
import mimetypes
import requests
from typing import Optional

# 15 秒以上は待たない
TIMEOUT = int(os.getenv("WP_API_TIMEOUT", "15"))


# ──────────────────────────────────────
# helper : 認証ヘッダー
# ──────────────────────────────────────
def _basic_auth_header(username: str, app_pass: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ──────────────────────────────────────
# helper : アイキャッチ画像アップロード
# ──────────────────────────────────────
def _upload_featured_image(site, image_url: str) -> Optional[int]:
    """
    image_url をダウンロード → 該当 WP の /wp-json/wp/v2/media へ POST
    成功時は media ID を返す。失敗時 None。
    """
    try:
        img_resp = requests.get(image_url, timeout=TIMEOUT)
        img_resp.raise_for_status()
    except Exception:
        return None

    filename = os.path.basename(image_url.split("?")[0])
    ctype = mimetypes.guess_type(filename)[0] or "image/jpeg"

    headers = {
        **_basic_auth_header(site.username, site.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }

    media_endpoint = f"{site.url.rstrip('/')}/wp-json/wp/v2/media"
    try:
        up = requests.post(
            media_endpoint,
            headers=headers,
            data=img_resp.content,
            timeout=TIMEOUT,
        )
        up.raise_for_status()
        return up.json().get("id")
    except Exception:
        return None


# ──────────────────────────────────────
# public : 記事投稿
# ──────────────────────────────────────
def post_to_wp(site, article) -> str:
    """
    Site インスタンスと Article インスタンスを受け取り、
    WordPress へ記事を「publish」ステータスで投稿。
    戻り値: 投稿先 URL (例外時は Exception を送出)
    """
    # ① featured image があれば media を先にアップロード
    featured_id = None
    if article.image_url:
        featured_id = _upload_featured_image(site, article.image_url)

    headers = _basic_auth_header(site.username, site.app_pass)
    api = f"{site.url.rstrip('/')}/wp-json/wp/v2/posts"

    payload = {
        "title":   article.title,
        "content": article.body,
        "status":  "publish",
    }
    if featured_id:
        payload["featured_media"] = featured_id

    resp = requests.post(api, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("link", "")
