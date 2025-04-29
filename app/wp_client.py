import base64
import mimetypes
import re
import requests
from datetime import datetime
from flask import current_app

# ──────────────────────────────────────────────
# WordPress 投稿ユーティリティ
# ──────────────────────────────────────────────
# ・Site インスタンスの .url, .username, .app_pass を使って動的に投稿
# ・本文の <h2>,<h3>,<p> に Tailwind CSS クラスを付与
# ・アイキャッチ画像アップロード対応
# ──────────────────────────────────────────────

# タイムアウト（秒）
TIMEOUT = 15

def make_auth_header(username: str, app_pass: str) -> dict[str, str]:
    """
    Basic 認証ヘッダーを作る
    """
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def _decorate_html(html: str) -> str:
    """
    本文中の <h2>, <h3>, <p> タグに装飾用の Tailwind CSS クラスを追加する
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
    アイキャッチ画像を /media エンドポイントへアップロードし、
    返却されたメディア ID を返す
    """
    current_app.logger.debug(f"Uploading image {image_path} to {api_media} as {username}")
    headers = make_auth_header(username, app_pass)
    mime_type, _ = mimetypes.guess_type(image_path)
    mime = mime_type or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, mime)}
        resp = requests.post(api_media, headers=headers, files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    media_id = resp.json().get("id")
    current_app.logger.debug(f"Uploaded image ID: {media_id}")
    return media_id

def post_to_wp(site, art) -> str:
    """
    記事を WordPress に投稿し、公開された記事の URL を返す。
    """
    # 動的エンドポイント
    base = site.url.rstrip("/")
    api_posts = f"{base}/wp-json/wp/v2/posts"
    api_media = f"{base}/wp-json/wp/v2/media"

    current_app.logger.debug(f"Posting article {art.id} to {api_posts} as {site.username}")
    headers = make_auth_header(site.username, site.app_pass) | {"Content-Type": "application/json"}
    current_app.logger.debug(f"Auth header: {headers['Authorization'][:30]}...")

    # 装飾付き本文
    body = _decorate_html(art.body or "")

    # アイキャッチ画像処理
    featured_id = None
    if getattr(art, "featured_image_url", None):
        featured_id = upload_featured_image(
            art.featured_image_url,
            api_media,
            site.username,
            site.app_pass
        )

    # リクエストデータ組み立て
    data: dict = {
        "title": art.title,
        "content": body,
        "status": "publish"
    }
    if featured_id:
        data["featured_media"] = featured_id
    if getattr(art, "categories", None):
        data["categories"] = art.categories

    # 投稿実行
    resp = requests.post(api_posts, headers=headers, json=data, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    link = result.get("link", "")
    current_app.logger.info(f"Posted article {art.id} to {link}")
    return link

# 互換性維持エイリアス
post_article = post_to_wp

# — 使用例 — (スクリプト単体実行時のみ)
if __name__ == "__main__":
    class DummySite:
        url = "https://business-search-abroad.com"
        username = "your-username"
        app_pass = "your-app-password"

    class DummyArt:
        id = 0
        title = "テスト投稿"
        body = "<p>テスト内容</p>"
        featured_image_url = None
        categories = []

    site = DummySite()
    art = DummyArt()
    print("Result URL:", post_to_wp(site, art))
