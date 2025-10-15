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
from typing import Dict, List, Optional, Tuple, Any
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

def _clean_meta(s: str, max_len: int | None = None) -> str:
    """
    メタ説明などを WP 送信用に軽くサニタイズ：
      - 両端空白除去
      - 改行・タブ・全角空白を半角スペースへ
      - 連続スペースを 1 つに圧縮
      - max_len があれば末尾を安全トリム
    """
    if not s:
        return ""
    txt = str(s).strip()
    txt = txt.replace("\u3000", " ")     # 全角スペース
    txt = txt.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # 連続空白を 1 つに
    while "  " in txt:
        txt = txt.replace("  ", " ")
    if max_len and len(txt) > max_len:
        txt = txt[:max_len].rstrip()
    return txt

def _truncate(s: str, n: int) -> str:
    return s if not s or len(s) <= n else s[:n].rstrip()

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

    # メタ説明（DBにあれば使用）— 改行/全角/連続空白をならし、180 文字に整形
    meta_desc = _clean_meta(art.meta_description or "", max_len=180)

    post_data = {
        "title": art.title,
        "content": f'<div class="ai-content">{_decorate_html(art.body)}</div>',
        "status": "publish",
    }
    # 保険として excerpt にもメタ説明を入れておく（多くのテーマ/SEOプラグインが拾う）
    if meta_desc:
        post_data["excerpt"] = meta_desc
    if featured_media_id:
        post_data["featured_media"] = featured_media_id

    try:
        response = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)
        if response.status_code == 201:
            art.status = "posted"
            art.posted_url = response.json().get("link")
            db.session.commit()
            current_app.logger.info(f"投稿成功: Article ID {art.id}, User: {art.user_id}, Site: {site.url} -> {art.posted_url}")
            # 可能なら SEO メタもプッシュ（失敗しても投稿は成功として進める）
            try:
                wp_id = int(response.json().get("id"))
                _push_seo_meta_to_wp(site, wp_id, art, meta_desc)
            except Exception as e:
                current_app.logger.warning(f"[WP-SEO] meta push skipped: {e}")
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

def _push_seo_meta_to_wp(site: Site, wp_post_id: int, art: Article, meta_desc: str = "") -> None:
    """
    WordPress 側に SEO メタをできる範囲で反映する補助処理。
    - すでに post 作成時に excerpt は送っているが、ここでも再度ベストエフォートで反映
    - Yoast / RankMath のメタキーにも試行（RESTで拒否される環境もあるため、失敗はログのみ）
    """
    import requests
    site_url = normalize_url(site.url)
    headers = _post_headers(site.username, site.app_pass, site_url)

    # 1) excerpt を再度同期（冪等）
    if meta_desc:
        try:
            resp = requests.post(
                f"{site_url}/wp-json/wp/v2/posts/{wp_post_id}",
                json={"excerpt": meta_desc},
                headers=headers,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            current_app.logger.info(f"[WP-SEO] excerpt sync skipped: {e}")

    # 2) 代表的SEOプラグインのメタキーにベストエフォートで書き込み
    #    Yoast:   _yoast_wpseo_title, _yoast_wpseo_metadesc
    #    RankMath: rank_math_title, rank_math_description
    # ※ REST で書けない設定のWPも多いので、失敗はログのみで継続
    # タイトルも軽整形＆60字トリム（一般的な推奨値）
    title_for_meta = _truncate(_clean_meta(art.title or art.keyword or ""), 60)
    meta_desc = _clean_meta(meta_desc or title_for_meta, max_len=180)
    meta_try_list = [
        {"_yoast_wpseo_title": title_for_meta},
        {"_yoast_wpseo_metadesc": meta_desc or title_for_meta},
        {"rank_math_title": title_for_meta},
        {"rank_math_description": meta_desc or title_for_meta},
    ]
    for meta_obj in meta_try_list:
        try:
            resp = requests.post(
                f"{site_url}/wp-json/wp/v2/posts/{wp_post_id}",
                json={"meta": meta_obj},
                headers=headers,
                timeout=TIMEOUT,
            )
            # 多くの環境で 400/403 が返る（RESTで未公開のmetaキー）。その場合は情報ログだけ。
            if not (200 <= resp.status_code < 300):
                current_app.logger.info(f"[WP-SEO] meta write {list(meta_obj.keys())[0]} -> {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            current_app.logger.info(f"[WP-SEO] meta write skipped ({list(meta_obj.keys())[0]}): {e}")

# =============================================================
# 🔸 NEW: Topicページ用の汎用投稿ヘルパ（Article不要）
# =============================================================
def post_topic_to_wp(
    site: Site,
    title: str,
    html: str,
    *,
    slug: Optional[str] = None,
    status: str = "publish",
    category_ids: Optional[List[int]] = None,
) -> Tuple[int, str]:
    """
    Topicページ（汎用HTML断片）を WordPress に投稿し、(post_id, link) を返す。
    - Article モデルに依存しない軽量版
    - slug を指定すると WP 側のスラッグに設定（将来の更新取得が容易）
    - category_ids は WordPress のカテゴリIDの配列（例：[12, 34]）。未指定ならカテゴリ付与なし。
    """
    site_url = normalize_url(site.url)
    url = f"{site_url}/wp-json/wp/v2/posts"
    headers = _post_headers(site.username, site.app_pass, site_url)

    post_data: Dict[str, Any] = {
        "title": title,
        "content": f'<div class="ai-content">{_decorate_html(html)}</div>',
        "status": status,
    }
    if slug:
        post_data["slug"] = slug
    # WordPress の REST は categories に「数値IDの配列」を要求
    if category_ids:
        post_data["categories"] = category_ids

    resp = requests.post(url, json=post_data, headers=headers, timeout=TIMEOUT)
    if resp.status_code == 201:
        data = resp.json()
        post_id = int(data.get("id"))
        link = data.get("link") or ""
        current_app.logger.info("[WP] topic posted: id=%s link=%s", post_id, link)
        return post_id, link
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    raise HTTPError(f"[WP] topic create failed status={resp.status_code} body={str(body)[:200]}")


# =============================================================
# 🔧 追加: 既存記事の「メタ説明だけ」を安全に更新するヘルパ
#   - 既存の post_to_wp() に一切影響を与えない“別入口”
#   - excerpt を確実に同期し、主要SEOプラグインの description キーにも
#     ベストエフォートで書き込む（失敗しても投稿は壊れない）
#   - 戻り値: True（いずれか成功）/ False（全部失敗）
# =============================================================
def update_post_meta(site: Site, wp_post_id: int, meta_description: str) -> bool:
    """
    既存WP投稿のメタ説明を更新する。
    - まず excerpt を更新（多くのテーマ/プラグインが拾う）
    - 次に Yoast / RankMath の description メタキーを書き込み（任意・失敗許容）
    """
    site_url = normalize_url(site.url)
    headers = _post_headers(site.username, site.app_pass, site_url)
    meta_desc = _clean_meta(meta_description or "", max_len=180)

    ok_any = False

    # 1) excerpt を更新（冪等）
    try:
        resp = requests.post(
            f"{site_url}/wp-json/wp/v2/posts/{wp_post_id}",
            json={"excerpt": meta_desc},
            headers=headers,
            timeout=TIMEOUT,
        )
        if 200 <= resp.status_code < 300:
            ok_any = True
        else:
            current_app.logger.info(
                "[WP-SEO] excerpt update failed %s: %s",
                resp.status_code, resp.text[:160]
            )
    except Exception as e:
        current_app.logger.info("[WP-SEO] excerpt update error: %s", e)

    # 2) 主要SEOプラグインの description メタキーをベストエフォートで更新
    #    タイトルは既存の投稿機能で十分なのでここでは触らない（安全最優先）
    for meta_obj in ({"_yoast_wpseo_metadesc": meta_desc},
                     {"rank_math_description": meta_desc}):
        try:
            resp = requests.post(
                f"{site_url}/wp-json/wp/v2/posts/{wp_post_id}",
                json={"meta": meta_obj},
                headers=headers,
                timeout=TIMEOUT,
            )
            if 200 <= resp.status_code < 300:
                ok_any = True
            else:
                # 多くの環境で 400/403 は通常（REST未公開メタ）
                current_app.logger.info(
                    "[WP-SEO] meta write %s -> %s: %s",
                    list(meta_obj.keys())[0], resp.status_code, resp.text[:160]
                )
        except Exception as e:
            current_app.logger.info(
                "[WP-SEO] meta write skipped (%s): %s",
                list(meta_obj.keys())[0], e
            )

    return ok_any