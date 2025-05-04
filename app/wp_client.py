import base64
import mimetypes
import os
import requests
from requests.exceptions import HTTPError
from flask import current_app
from .models import Site, Article

# タイムアウト（秒）
TIMEOUT = 30

# Basic認証ヘッダーを作成する関数（User-Agent追加済み）
def _basic_auth_header(username: str, app_pass: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
    }

# 画像をWordPressにアップロードする関数
def upload_image_to_wp(site_url: str, image_path: str, username: str, app_pass: str):
    url = f"{site_url}/wp-json/wp/v2/media"
    headers = _basic_auth_header(username, app_pass)

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg'

    with open(image_path, 'rb') as image_file:
        files = {
            'file': (os.path.basename(image_path), image_file, mime_type)
        }
        response = requests.post(url, headers=headers, files=files, timeout=TIMEOUT)

    if response.status_code == 201:
        data = response.json()
        return data["id"], data["source_url"]
    else:
        print(f"詳細なエラー: {response.json()}")
        raise HTTPError(f"画像のアップロードに失敗しました: {response.status_code}, {response.text}")

# 投稿を行うメイン関数（統一版）
def post_to_wp(site: Site, art: Article) -> str:
    url = f"{site.url}/wp-json/wp/v2/posts"
    headers = _basic_auth_header(site.username, site.app_pass)

    featured_media_id = None

    if art.image_url and art.image_url.startswith("http"):
        try:
            response = requests.get(art.image_url, timeout=10)
            ext = os.path.splitext(art.image_url)[-1].split("?")[0] or ".jpg"
            temp_path = f"temp_featured_image{ext}"
            with open(temp_path, "wb") as f:
                f.write(response.content)

            featured_media_id, uploaded_url = upload_image_to_wp(
                site.url, temp_path, site.username, site.app_pass
            )

            art.featured_image = uploaded_url
            os.remove(temp_path)

        except Exception as e:
            current_app.logger.warning(f"アイキャッチ画像のアップロード失敗: {e}")

    post_data = {
        "title": art.title,
        "content": art.body,
        "status": "publish",
    }
    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    response = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)
    if response.status_code == 201:
        return response.json().get("link") or "success"
    else:
        raise HTTPError(f"記事の作成に失敗: {response.status_code}, {response.text}")

# デザイン装飾用
def _decorate_html(content: str) -> str:
    content = content.replace('<h2>', '<h2 style="font-size: 24px; color: blue;">')
    content = content.replace('<h3>', '<h3 style="font-size: 20px; color: green;">')
    content = content.replace('<p>',  '<p style="font-family: Arial, sans-serif; line-height: 1.6;">')
    return content
