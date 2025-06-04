import base64
import mimetypes
import os
import requests
import time
from . import db
from requests.exceptions import HTTPError
from flask import current_app
from .models import Site, Article

# タイムアウト（秒）
TIMEOUT = 30

# URL正規化
def normalize_url(url: str) -> str:
    return url.rstrip('/')

# 投稿用のヘッダー作成（application/json 用）
def _post_headers(username: str, app_pass: str, site_url: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0',
        'Referer': f'{site_url}/wp-admin',
        'Origin': site_url,
        'Accept': '*/*, application/json',
    }

# 画像アップロード用のヘッダー作成
def _upload_headers(username: str, app_pass: str, site_url: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'User-Agent': 'Mozilla/5.0',
        'Referer': f'{site_url}/wp-admin',
        'Accept': '*/*',
    }

# ✅ 画像をWordPressにアップロードし、記事タイトルをtitle/alt_textとして設定
def upload_image_to_wp(site_url: str, image_path: str, username: str, app_pass: str, image_title: str = ""):
    site_url = normalize_url(site_url)
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
        media_id = data["id"]
        source_url = data["source_url"]

        # ✅ メタ情報（title, alt_text）を記事タイトルで上書き
        if image_title:
            patch_url = f"{site_url}/wp-json/wp/v2/media/{media_id}"
            patch_data = {
                "title": image_title,
                "alt_text": image_title,
                "caption": "",
                "description": ""
            }
            patch_headers = _post_headers(username, app_pass, site_url)
            try:
                patch_res = requests.post(patch_url, headers=patch_headers, json=patch_data, timeout=TIMEOUT)
                if patch_res.status_code not in [200, 201]:
                    current_app.logger.warning(f"画像メタ情報の更新に失敗: {patch_res.status_code}")
            except Exception as e:
                current_app.logger.warning(f"画像メタ情報のPATCHエラー: {e}")

        return media_id, source_url
    else:
        try:
            error = response.json()
        except Exception:
            error = response.text
        current_app.logger.warning(f"画像のアップロードに失敗: {response.status_code}, {error}")
        raise HTTPError(f"画像のアップロードに失敗しました: {response.status_code}, {error}")

# WordPress投稿処理（画像アップロード処理を拡張）
def post_to_wp(site: Site, art: Article) -> str:
    # ✅ すでに投稿済みかどうかチェック（重要）
    if art.status == "posted" and art.posted_url:
        current_app.logger.info(f"[スキップ] すでに投稿済み: Article ID {art.id}")
        return art.posted_url or "already posted"

    site_url = normalize_url(site.url)
    url = f"{site_url}/wp-json/wp/v2/posts"
    headers = _post_headers(site.username, site.app_pass, site_url)

    featured_media_id = None

    if art.image_url:
        try:
            if art.image_url.startswith("/static/images/"):
                image_path = os.path.join("app", art.image_url.lstrip("/"))
                featured_media_id, uploaded_url = upload_image_to_wp(
                    site_url, image_path, site.username, site.app_pass, image_title=art.title
                )
                art.featured_image = uploaded_url

            elif art.image_url.startswith("http"):
                response = requests.get(art.image_url, timeout=10)
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise ValueError(f"取得先が画像ではありません: {content_type}")
                ext = os.path.splitext(art.image_url)[-1].split("?")[0]
                if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
                temp_path = f"temp_featured_image{ext}"
                with open(temp_path, "wb") as f:
                    f.write(response.content)

                featured_media_id, uploaded_url = upload_image_to_wp(
                    site_url, temp_path, site.username, site.app_pass, image_title=art.title
                )
                art.featured_image = uploaded_url
                os.remove(temp_path)

        except Exception as e:
            current_app.logger.warning(f"アイキャッチ画像のアップロード失敗: {e}")

    post_data = {
        "title": art.title,
        "content": f'<div class="ai-content">{_decorate_html(art.body)}</div>',
        "status": "publish",
    }
    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    for attempt in range(3):
        try:
            response = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)
            if response.status_code == 201:
                art.status = "posted"
                art.posted_url = response.json().get("link")
                db.session.commit()
                return response.json().get("link") or "success"
            else:
                raise HTTPError(f"ステータスコード {response.status_code}")
        except Exception as e:
            if attempt < 2:
                current_app.logger.warning(f"[{attempt+1}回目] 投稿リトライ中: {e}")
                time.sleep(2)
            else:
                try:
                    error = response.json()
                except Exception:
                    error = response.text
                current_app.logger.error(f"記事の作成に失敗: {response.status_code}, {error}")
                raise HTTPError(f"記事の作成に失敗: {response.status_code}, {error}")

# デザイン調整
def _decorate_html(content: str) -> str:
    content = content.replace('<h2>', '<h2 class="ai-h2">')
    content = content.replace('<h3>', '<h3 class="ai-h3">')
    content = content.replace('<p>', '<p class="ai-p">')
    return content
