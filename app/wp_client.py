import base64
import mimetypes
import re
import requests
from datetime import datetime

# ──────────────────────────────────────────────
# WordPress 投稿ユーティリティ
# ──────────────────────────────────────────────
# ・Site モデルの .url/.username/.app_pass を使って動的に投稿
# ・本文の <h2>,<h3>,<p> に Tailwind CSS クラスを付与
# ・アイキャッチ画像アップロード対応
# ──────────────────────────────────────────────

# タイムアウト（秒）
TIMEOUT = 15


def make_auth_header(username: str, app_pass: str) -> dict[str, str]:
    """
    Basic 認証ヘッダーを作成
    """
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _decorate_html(html: str) -> str:
    """
    本文中の <h2>, <h3>, <p> タグに Tailwind 用クラスを追加
    """
    html = re.sub(r'<h2(?![^>]*class=)', '<h2 class="text-xl font-bold mt-4 mb-2"', html)
    html = re.sub(r'<h3(?![^>]*class=)', '<h3 class="text-lg font-semibold mt-3 mb-1"', html)
    html = re.sub(r'<p(?![^>]*class=)', '<p class="mb-4 leading-relaxed"', html)
    return html


def upload_featured_image(
    image_path: str,
    api_media: str,
    username: str,
    app_pass: str
) -> int:
    """
    アイキャッチ画像を WordPress の /media エンドポイントへアップロードし、
    返却されたメディア ID を返す
    """
    headers = make_auth_header(username, app_pass)
    mime_type, _ = mimetypes.guess_type(image_path)
    mime = mime_type or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, mime)}
        resp = requests.post(api_media, headers=headers, files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["id"]


def post_to_wp(
    site,   # Site モデルインスタンス: .url, .username, .app_pass
    art     # Article モデルインスタンス: .title, .body, .featured_image_url, .categories
) -> str:
    """
    Article を WordPress に投稿し、公開 URL を返す
    """
    # 1) エンドポイントを動的に組み立て
    base_url = site.url.rstrip("/")
    api_posts = f"{base_url}/wp-json/wp/v2/posts"
    api_media = f"{base_url}/wp-json/wp/v2/media"

    # 2) 本文に CSS クラスを付与
    body = _decorate_html(art.body or "")

    # 3) アイキャッチ画像アップロード
    featured_id = None
    if getattr(art, "featured_image_url", None):
        featured_id = upload_featured_image(
            art.featured_image_url,
            api_media,
            site.username,
            site.app_pass
        )

    # 4) 投稿データ組み立て
    headers = make_auth_header(site.username, site.app_pass) | {"Content-Type": "application/json"}
    data = {
        "title": art.title,
        "content": body,
        "status": "publish"
    }
    if featured_id:
        data["featured_media"] = featured_id
    if getattr(art, "categories", None):
        data["categories"] = art.categories

    # 5) 投稿実行
    resp = requests.post(api_posts, headers=headers, json=data, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()

    # 6) 公開 URL を返す
    return result.get("link", "")

# 互換性エイリアス
post_article = post_to_wp
