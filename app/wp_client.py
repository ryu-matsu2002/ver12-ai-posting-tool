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
# ・リモート URL／ローカルファイル両対応でアイキャッチ画像をアップロード
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
    （image_path が URL なら先にダウンロード、ファイルパスなら直接 open） 
    """
    current_app.logger.debug(f"Uploading image [{image_path}] to [{api_media}] as [{username}]")
    headers = make_auth_header(username, app_pass)

    # MIME タイプ決定
    mime_type, _ = mimetypes.guess_type(image_path)
    mime = mime_type or "application/octet-stream"

    # リモート URL の場合は先に取得
    if image_path.startswith("http://") or image_path.startswith("https://"):
        resp = requests.get(image_path, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.content
        filename = image_path.rsplit("/", 1)[-1] or "upload.jpg"
        files = {"file": (filename, data, mime)}
    else:
        # ローカルファイル
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, mime)}

    resp = requests.post(api_media, headers=headers, files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    media_id = resp.json().get("id")
    current_app.logger.debug(f"Uploaded media ID: {media_id}")
    return media_id

def post_to_wp(site, art) -> str:
    """
    記事を WordPress に投稿し、公開された記事の URL を返す。
    """
    # ─ 動的エンドポイント生成 ─
    base = site.url.rstrip("/")
    api_posts = f"{base}/wp-json/wp/v2/posts"
    api_media = f"{base}/wp-json/wp/v2/media"

    current_app.logger.debug(f"Posting Article#{art.id} to {api_posts} as {site.username}")
    headers = make_auth_header(site.username, site.app_pass) | {"Content-Type": "application/json"}
    # デバッグ追加
    current_app.logger.debug(f"AUTHORIZATION header being sent: {headers['Authorization']}")
    current_app.logger.debug(f"Auth header (truncated): {headers['Authorization'][:30]}...")

    # 本文装飾
    body = _decorate_html(art.body or "")

    # アイキャッチ画像 ID を取得
    featured_id = None
    img_url = getattr(art, "featured_image_url", None) or getattr(art, "image_url", None)
    if img_url:
        try:
            featured_id = upload_featured_image(img_url, api_media, site.username, site.app_pass)
        except Exception as e:
            current_app.logger.warning(f"Failed uploading featured image: {e}")

    # 投稿 payload
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
    current_app.logger.info(f"Posted Article#{art.id} to {link}")
    return link

# 互換性維持エイリアス
post_article = post_to_wp

# — スクリプト単体実行テスト用 —
if __name__ == "__main__":
    class DummySite:
        url = "https://business-search-abroad.com"
        username = "your-username"
        app_pass = "your-app-password"

    class DummyArt:
        id = 0
        title = "テスト投稿"
        body = "<h2>見出し</h2><p>本文</p>"
        featured_image_url = "https://via.placeholder.com/600x400"
        categories = []

    site = DummySite()
    art = DummyArt()
    print("Result URL:", post_to_wp(site, art))
