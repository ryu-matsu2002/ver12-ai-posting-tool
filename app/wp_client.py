import base64
import mimetypes
import os
import requests
import time
from . import db
from requests.exceptions import HTTPError
from flask import current_app
from .models import Site, Article, Error, InternalSeoConfig
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin




# タイムアウト（秒）
TIMEOUT = 30
RETRY_BACKOFF = [1, 2, 4, 8]  # 内部SEO用の再試行待ち（秒）

# --- ① ブラウザを装う汎用 UA --------------------------------------------
UA_FAKE = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
# ---------------------------------------------------------------------------

# URL正規化
def normalize_url(url: str) -> str:
    return url.rstrip('/')

# 投稿用のヘッダー作成（application/json 用）
def _post_headers(username: str, app_pass: str, site_url: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json',
        'User-Agent': UA_FAKE, 
        'Referer': f'{site_url}/wp-admin',
        'Origin': site_url,
        'Accept': '*/*, application/json',
    }

def _get_headers(username: str, app_pass: str, site_url: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'User-Agent': UA_FAKE,
        'Referer': f'{site_url}/wp-admin',
        'Origin': site_url,
        'Accept': 'application/json',
    }

# 画像アップロード用のヘッダー作成
def _upload_headers(username: str, app_pass: str, site_url: str) -> dict:
    token = base64.b64encode(f'{username}:{app_pass}'.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {token}',
        'User-Agent': UA_FAKE, 
        'Referer': f'{site_url}/wp-admin',
        'Accept': '*/*',
    }

# =========================
# 内部SEO 用ユーティリティ
# =========================

def _wp_posts_endpoint(base_url: str) -> str:
    # 例: https://example.com/wp-json/wp/v2/posts
    base = normalize_url(base_url) + "/"
    return urljoin(base, "wp-json/wp/v2/posts")

def _wp_single_post_endpoint(base_url: str, post_id: int) -> str:
    return urljoin(_wp_posts_endpoint(base_url).rstrip("/") + "/", str(post_id))

def _request_with_retry(method: str, url: str, headers: Dict[str, str], params=None, json_body=None, timeout=TIMEOUT) -> requests.Response:
    last_exc = None
    for attempt, backoff in enumerate([0] + RETRY_BACKOFF):
        if attempt:
            current_app.logger.warning("[WP] retrying %s %s (attempt=%s)", method, url, attempt + 1)
            time.sleep(backoff)
        try:
            resp = requests.request(method.upper(), url, headers=headers, params=params, json=json_body, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"Transient HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            continue
    raise last_exc or RuntimeError("WP request failed after retries")

def _ensure_internal_seo_config(site_id: int) -> InternalSeoConfig:
    cfg = InternalSeoConfig.query.filter_by(site_id=site_id).one_or_none()
    if cfg:
        return cfg
    cfg = InternalSeoConfig(site_id=site_id)
    db.session.add(cfg)
    db.session.commit()
    current_app.logger.info("[InternalSEO] Created default config site_id=%s", site_id)
    return cfg

def _rate_limit(site: Site) -> None:
    cfg = _ensure_internal_seo_config(site.id)
    per_min = max(1, int(cfg.rate_limit_per_minute or 10))
    sleep_sec = max(0.0, 60.0 / float(per_min))
    if sleep_sec > 0:
        time.sleep(min(sleep_sec, 2.0))  # 2秒上限で軽く抑制

@dataclass
class WPPost:
    id: int
    link: str
    title_html: str
    content_html: str
    modified_gmt: Optional[str]
    status: str
    slug: Optional[str] = None

    def excluded_by_topic(self, exclude_topic: bool) -> bool:
        if not exclude_topic:
            return False
        return "topic" in (self.link or "").lower()

# -------------------------
# 取得（ページング対応）
# -------------------------
def fetch_posts_paged(
    site: Site,
    page: int = 1,
    per_page: int = 100,
    status: str = "publish",
    after_gmt: Optional[str] = None,
) -> Tuple[List[WPPost], int]:
    """
    公開記事を1ページ分取得する（内部SEO用）。
    - URLに 'topic' を含む記事は除外（configでONのとき）
    戻り: (posts, total_pages)
    """
    _rate_limit(site)
    cfg = _ensure_internal_seo_config(site.id)
    site_url = normalize_url(site.url)
    headers = _get_headers(site.username, site.app_pass, site_url)
    params = {
        "status": status,
        "page": page,
        "per_page": min(max(per_page, 1), 100),
        "context": "edit",
        "_fields": "id,link,title,content,modified_gmt,slug,status",
    }
    if after_gmt:
        params["after"] = after_gmt

    url = _wp_posts_endpoint(site_url)
    resp = _request_with_retry("GET", url, headers, params=params)
    total_pages = int(resp.headers.get("X-WP-TotalPages", "1") or "1")

    posts: List[WPPost] = []
    for item in resp.json():
        p = WPPost(
            id=int(item.get("id")),
            link=item.get("link") or "",
            title_html=(item.get("title") or {}).get("rendered") or "",
            content_html=(item.get("content") or {}).get("rendered") or "",
            modified_gmt=item.get("modified_gmt"),
            status=item.get("status") or "publish",
            slug=item.get("slug"),
        )
        if cfg.exclude_topic_in_url and p.excluded_by_topic(True):
            continue
        posts.append(p)
    return posts, total_pages

# -------------------------
# 単一記事の取得
# -------------------------
def fetch_single_post(site: Site, post_id: int) -> Optional[WPPost]:
    _rate_limit(site)
    cfg = _ensure_internal_seo_config(site.id)
    site_url = normalize_url(site.url)
    headers = _get_headers(site.username, site.app_pass, site_url)
    params = {"context": "edit", "_fields": "id,link,title,content,modified_gmt,slug,status"}
    url = _wp_single_post_endpoint(site_url, post_id)
    try:
        resp = _request_with_retry("GET", url, headers, params=params)
        item = resp.json()
        p = WPPost(
            id=int(item.get("id")),
            link=item.get("link") or "",
            title_html=(item.get("title") or {}).get("rendered") or "",
            content_html=(item.get("content") or {}).get("rendered") or "",
            modified_gmt=item.get("modified_gmt"),
            status=item.get("status") or "publish",
            slug=item.get("slug"),
        )
        if cfg.exclude_topic_in_url and p.excluded_by_topic(True):
            return None
        return p
    except Exception as e:
        current_app.logger.warning("[WP] fetch_single_post failed site_id=%s post_id=%s: %s", site.id, post_id, e)
        return None

# -------------------------
# 本文の差分更新（安全パッチ）
# -------------------------
def update_post_content(site: Site, post_id: int, new_html: str) -> bool:
    """
    内部SEOで加工した本文HTMLをWordPressに反映（差分は上位層で生成）。
    WordPressの更新は POST /wp-json/wp/v2/posts/{id} を使用。
    """
    _rate_limit(site)
    site_url = normalize_url(site.url)
    headers = _post_headers(site.username, site.app_pass, site_url)
    url = _wp_single_post_endpoint(site_url, post_id)
    payload = {"content": new_html}
    try:
        resp = _request_with_retry("POST", url, headers, json_body=payload)
        ok = 200 <= resp.status_code < 300
        if not ok:
            current_app.logger.error("[WP] update content failed status=%s body=%s", resp.status_code, resp.text[:200])
        return ok
    except Exception as e:
        current_app.logger.error("[WP] update_post_content error site_id=%s post_id=%s: %s", site.id, post_id, e)
        return False

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

def log_error_to_db(article_id, user_id, site_id, error_message):
    try:
        error = Error(
            article_id=article_id,
            user_id=user_id,
            site_id=site_id,
            error_message=error_message,
            created_at=datetime.utcnow()
        )
        db.session.add(error)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"エラー情報の保存失敗: {e}")

# WordPress投稿処理（画像アップロード処理を拡張）
def post_to_wp(site: Site, art: Article) -> str:
    # ✅ すでに投稿済みかどうかチェック（重要）
    if art.status == "posted" and art.posted_url:
        current_app.logger.info(f"[スキップ] すでに投稿済み: Article ID {art.id}, User: {art.user_id}, Site: {site.url}")
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
            current_app.logger.warning(f"アイキャッチ画像のアップロード失敗: Article ID {art.id}, User: {art.user_id}, Site: {site.url}, エラー: {e}")

    post_data = {
        "title": art.title,
        "content": f'<div class="ai-content">{_decorate_html(art.body)}</div>',
        "status": "publish",
    }
    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    try:
        response = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)
        if response.status_code == 201:
            art.status = "posted"
            art.posted_url = response.json().get("link")
            db.session.commit()
            current_app.logger.info(f"投稿成功: Article ID {art.id}, User: {art.user_id}, Site: {site.url} -> {art.posted_url}")
            return art.posted_url or "success"
        else:
            raise HTTPError(f"ステータスコード {response.status_code}")
    except Exception as e:
        current_app.logger.error(f"記事の作成に失敗: Article ID {art.id}, User: {art.user_id}, Site: {site.url}, エラー: {str(e)}")
        # 投稿失敗時にステータスを "error" に変更
        art.status = "error"
        db.session.commit()
        return f"Error: {str(e)}"


# デザイン調整
def _decorate_html(content: str) -> str:
    content = content.replace('<h2>', '<h2 class="ai-h2">')
    content = content.replace('<h3>', '<h3 class="ai-h3">')
    content = content.replace('<p>', '<p class="ai-p">')
    return content
