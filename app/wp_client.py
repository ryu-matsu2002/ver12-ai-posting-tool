import base64
import requests
from datetime import datetime

# ———— 設定 ————
WP_SITE      = "https://your-site.com"                         # あなたのサイト URL
API_POSTS    = f"{WP_SITE}/wp-json/wp/v2/posts"
API_MEDIA    = f"{WP_SITE}/wp-json/wp/v2/media"
USER         = "your-username"                                 # WP 管理ユーザー名
APP_PASSWORD = "your-application-password"                     # 発行したアプリケーションパスワード
TIMEOUT      = 15                                              # タイムアウト秒
# ——————————

def make_auth_header(user: str, pwd: str) -> dict[str, str]:
    """
    Basic 認証ヘッダーを作る
    """
    token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def upload_featured_image(image_path: str) -> int:
    """
    アイキャッチ画像を /media エンドポイントへアップロードし、
    返却されたメディア ID を返す
    """
    headers = make_auth_header(USER, APP_PASSWORD)
    # ファイルの MIME タイプ推定
    mime = requests.utils.guess_filename(image_path) or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, mime)}
        resp = requests.post(API_MEDIA, headers=headers, files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["id"]

def post_to_wp(
    site,      # サイトモデル（.url/.username/.app_pass を持つ）
    art        # 記事モデル（.title/.body/.featured_image_url/.categories を持つ）
) -> str:
    """
    記事を WordPress に投稿し、公開された記事の URL を返す。
    既存の post_article を post_to_wp にリネームしてエクスポート。
    """
    # もしアイキャッチ画像 URL があれば先にアップロードして ID を取得
    featured_id = None
    if art.featured_image_url:
        featured_id = upload_featured_image(art.featured_image_url)

    # 本文とタイトルを組み立て
    headers = make_auth_header(USER, APP_PASSWORD) | {"Content-Type": "application/json"}
    data = {
        "title": art.title,
        "content": art.body,
        "status": "publish"  # 即時公開。予約投稿にする場合は "future" と date フィールドを追加。
    }
    if featured_id:
        data["featured_media"] = featured_id
    if getattr(art, "categories", None):
        data["categories"] = art.categories

    # 投稿
    resp = requests.post(API_POSTS, headers=headers, json=data, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()

    # 公開 URL を返す
    return result.get("link", "")

# alias: もし他コードが post_article を呼んでいる場合の互換性維持
post_article = post_to_wp

# — 使用例 — (スクリプト単体実行時のみ)
if __name__ == "__main__":
    class DummyArt:
        title = "こんにちは世界"
        body = "<p>自動投稿テストです。</p>"
        featured_image_url = "path/to/eyecatch.jpg"
        categories = [2, 5]

    art = DummyArt()
    url = post_to_wp(None, art)
    print("投稿完了:", url)
