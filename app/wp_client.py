import base64
import mimetypes
import os
import requests
from requests.exceptions import HTTPError
from flask import current_app
from .models import Site, Article

# タイムアウト（秒）
TIMEOUT = 30

# 投稿用のヘッダー作成（application/json 用）
def _post_headers(username: str, app_pass: str, referer: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': referer,
        'Accept': 'application/json',
        'Connection': 'keep-alive'
    }

# 画像アップロード用のヘッダー作成（Content-Typeなし）
def _upload_headers(username: str, app_pass: str, referer: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': referer,
        'Accept': '*/*',
        'Connection': 'keep-alive'
    }

# 画像をWordPressにアップロードする関数
def upload_image_to_wp(site_url: str, image_path: str, username: str, app_pass: str):
    url = f"{site_url}/wp-json/wp/v2/media"
    headers = _upload_headers(username, app_pass, site_url)

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
        try:
            error = response.json()
        except Exception:
            error = response.text
        current_app.logger.warning(f"画像のアップロードに失敗: {response.status_code}, {error}")
        current_app.logger.warning(f"画像アップ失敗ヘッダー: {response.headers}")
        raise HTTPError(f"画像のアップロードに失敗しました: {response.status_code}, {error}")

# 投稿を行うメイン関数
def post_to_wp(site: Site, art: Article) -> str:
    url = f"{site.url}/wp-json/wp/v2/posts"
    headers = _post_headers(site.username, site.app_pass, site.url)

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
        "content": _decorate_html(art.body),
        "status": "publish",
    }
    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    response = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)

    if response.status_code == 201:
        return response.json().get("link") or "success"
    else:
        try:
            error = response.json()
        except Exception:
            error = response.text

        if response.status_code == 401:
            code = error.get("code")
            if code == "rest_cannot_create":
                raise HTTPError(
                    "401エラー: 投稿権限がないか、Basic認証がブロックされています。"
                    "\n以下を確認してください：\n"
                    "- サイトURLが https:// で始まっているか\n"
                    "- ユーザーが 投稿者 以上の権限を持っているか\n"
                    "- サーバーの .htaccess に Authorization ヘッダーの許可設定があるか"
                )

        current_app.logger.error(f"記事の作成に失敗: {response.status_code}, {error}")
        raise HTTPError(f"記事の作成に失敗: {response.status_code}, {error}")

# デザイン装飾用
def _decorate_html(content: str) -> str:
    content = content.replace('<h2>', '<h2 style="font-size: 24px; color: blue;">')
    content = content.replace('<h3>', '<h3 style="font-size: 20px; color: green;">')
    content = content.replace('<p>',  '<p style="font-family: Arial, sans-serif; line-height: 1.6;">')
    return content
