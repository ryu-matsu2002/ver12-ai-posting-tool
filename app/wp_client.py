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

# Basic 認証ヘッダーを作る
def make_auth_header(user: str, pwd: str) -> dict[str,str]:
    token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

# 1) アイキャッチ画像をアップロードし、メディア ID を取得
def upload_featured_image(image_path: str) -> int:
    headers = make_auth_header(USER, APP_PASSWORD)
    # ファイルの MIME タイプ推定
    mime = requests.utils.guess_filename(image_path) or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, mime)}
        resp = requests.post(API_MEDIA, headers=headers, files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["id"]

# 2) 記事を投稿（即時公開 or 予約投稿）
def post_article(
    title: str,
    content: str,
    featured_media_id: int | None = None,
    categories: list[int] | None = None,
    publish_datetime: datetime | None = None
) -> dict:
    headers = make_auth_header(USER, APP_PASSWORD) | {"Content-Type": "application/json"}
    data = {
        "title": title,
        "content": content,
        "status": "publish" if publish_datetime is None else "future"
    }
    if featured_media_id:
        data["featured_media"] = featured_media_id
    if categories:
        data["categories"] = categories
    if publish_datetime:
        # ISO 8601 形式でスケジュール
        data["date"] = publish_datetime.isoformat()
    resp = requests.post(API_POSTS, headers=headers, json=data, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

# — 使用例 —
if __name__ == "__main__":
    # 画像アップロード（アイキャッチ）
    img_id = upload_featured_image("path/to/eyecatch.jpg")
    # 記事投稿（即時公開）
    result = post_article(
        title="こんにちは世界",
        content="<p>自動投稿テストです。</p>",
        featured_media_id=img_id,
        categories=[2,5]
    )
    print("投稿完了:", result["link"])
