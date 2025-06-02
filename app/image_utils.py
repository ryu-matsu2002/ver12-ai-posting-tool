from __future__ import annotations
import os, random, time, logging, requests, re
from datetime import datetime
from typing import List
from flask import current_app
from .models import Article  # ✅ DB参照用
from werkzeug.utils import secure_filename

# ───── 設定 ─────
ROOT_URL            = os.getenv("APP_ROOT_URL", "https://your-domain.com")
PIXABAY_API_KEY     = os.getenv("PIXABAY_API_KEY", "")
PIXABAY_TIMEOUT     = 5
MAX_PER_PAGE        = 30
RECENTLY_USED_TTL   = int(os.getenv("IMAGE_CACHE_TTL", "86400"))  # 24h
DEFAULT_IMAGE_PATH  = os.getenv("DEFAULT_IMAGE_URL", "/static/images/default-thumb.jpg")
DEFAULT_IMAGE_URL   = (
    DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH.startswith("http")
    else f"{ROOT_URL}{DEFAULT_IMAGE_PATH}"
)

IMAGE_SAVE_DIR = os.path.join("app", "static", "images")
IMAGE_URL_PREFIX = "/static/images"

_BUSINESS_TAGS = {
    "money","business","office","finance","analysis",
    "marketing","startup","strategy","computer",
    "statistics","success"
}

_used_image_urls: dict[str, float] = {}

def _is_recently_used(url: str, ttl: int = RECENTLY_USED_TTL) -> bool:
    ts = _used_image_urls.get(url)
    return ts is not None and (time.time() - ts) < ttl

def _mark_used(url: str) -> None:
    _used_image_urls[url] = time.time()

def _is_image_url(url: str) -> bool:
    if not url or url.strip() in ["", "None"]:
        return False
    if url.startswith("/static/images/"):
        filename = os.path.basename(url)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return False
        local_path = os.path.join("app", "static", "images", filename)
        return os.path.exists(local_path) and os.path.getsize(local_path) > 0
    if url.startswith("http"):
        try:
            r = requests.head(url, timeout=3, allow_redirects=True)
            content_type = r.headers.get("Content-Type", "").lower()
            return "image" in content_type
        except Exception as e:
            logging.warning(f"[画像URL確認失敗] {url}: {e}")
            return False
    return False

def _sanitize_filename(title: str) -> str:
    clean_title = title.replace("/", "_").replace("\\", "_").strip()
    today = datetime.now().strftime("%Y%m%d")
    return f"{clean_title}-{today}.jpg"

def _search_pixabay(query: str, per_page: int = MAX_PER_PAGE) -> List[dict]:
    if not PIXABAY_API_KEY or not query:
        return []
    query = query.replace("　", " ").strip()
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": per_page,
        "safesearch": "true",
    }
    try:
        r = requests.get("https://pixabay.com/api/", params=params, timeout=PIXABAY_TIMEOUT)
        if r.status_code == 400:
            logging.warning("Pixabay 400 for %s – fallthrough", query)
            return []
        r.raise_for_status()
        return r.json().get("hits", [])
    except Exception as e:
        logging.debug("Pixabay error (%s): %s", query, e)
        return []

def _score(hit: dict, kw_set: set[str]) -> int:
    tags = {t.strip().lower() for t in hit.get("tags", "").split(",")}
    base = sum(1 for kw in kw_set if kw in tags)
    bonus = 2 if tags & _BUSINESS_TAGS else 0
    return base + bonus

def _valid_dim(hit: dict) -> bool:
    w, h = hit.get("imageWidth", 0), hit.get("imageHeight", 1)
    if w <= 0 or h <= 0:
        return False
    ratio = w / h
    return 0.5 <= ratio <= 3.0

def _pick_pixabay(hits: List[dict], keywords: List[str], exclude_urls: set[str]) -> str:
    if not hits:
        return DEFAULT_IMAGE_URL
    kw_set = {k.lower() for k in keywords}
    top = sorted(hits, key=lambda h: _score(h, kw_set), reverse=True)[:10]
    random.shuffle(top)

    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and url not in exclude_urls and not _is_recently_used(url) and _valid_dim(h) and _is_image_url(url):
            _mark_used(url)
            return url

    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and url not in exclude_urls and not _is_recently_used(url) and _is_image_url(url):
            _mark_used(url)
            return url

    url = top[0].get("largeImageURL") or top[0].get("webformatURL") or DEFAULT_IMAGE_URL
    _mark_used(url)
    return url

def _unsplash_src(query: str) -> str:
    words = query.split()[:6]
    short = " ".join(words)[:120]
    q = requests.utils.quote(short)
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

def _download_and_save_image(image_url: str, title: str) -> str:
    try:
        filename = _sanitize_filename(title)
        local_path = os.path.join(IMAGE_SAVE_DIR, filename)
        if not os.path.exists(IMAGE_SAVE_DIR):
            os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        if not os.path.exists(local_path):
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        return f"{IMAGE_URL_PREFIX}/{filename}"
    except Exception as e:
        logging.error(f"[画像保存失敗] {image_url}: {e}")
        return DEFAULT_IMAGE_URL

# ─────────────────────────────────────────────
# 🔧 Public API
# ─────────────────────────────────────────────
def fetch_featured_image(query: str, title: str = "", body: str = "") -> str:
    def extract_keywords_from_body(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)[:500]
        words = re.split(r"[　\s]+", text)
        stopwords = {"の", "に", "で", "を", "と", "が", "は", "から", "ため", "こと", "について", "方法", "おすすめ", "完全ガイド"}
        return " ".join([w for w in words if w and w not in stopwords])[:120]

    def clean_and_translate(query: str) -> str:
        jp_to_en = {
            "ピラティス": "pilates", "腰痛": "back pain", "改善": "improvement", "福岡": "fukuoka",
            "英語": "english", "副業": "side job", "ブログ": "blog", "ビジネス": "business",
            "旅行": "travel", "脱毛": "hair removal", "メイク": "makeup", "占い": "fortune telling"
        }
        keywords = [w for w in re.split(r"[\\s　]+", query) if w]
        translated = [jp_to_en.get(w, w) for w in keywords]
        return " ".join(translated[:6])  # ✅ 最大6語に制限

    try:
        body_query = extract_keywords_from_body(body or "")
        base_query = clean_and_translate(f"{query} {body_query}")
        keywords = base_query.split()

        with current_app.app_context():
            recent_urls = {
                a.image_url for a in Article.query
                .filter(Article.image_url != None)
                .order_by(Article.created_at.desc())
                .limit(100)
            }

        hits = _search_pixabay(base_query)
        url = _pick_pixabay(hits, keywords, exclude_urls=recent_urls)
        if url and url.lower().endswith(('.jpg', '.jpeg', '.png')):
            return _download_and_save_image(url, title or query)

        fallback_query = f"{base_query} woman person lifestyle"
        hits = _search_pixabay(fallback_query)
        url = _pick_pixabay(hits, keywords + ["woman", "person", "lifestyle"], exclude_urls=recent_urls)
        if url and url.lower().endswith(('.jpg', '.jpeg', '.png')):
            return _download_and_save_image(url, title or query)

        return _unsplash_src(base_query)

    except Exception as e:
        logging.error("fetch_featured_image fatal: %s", e)
        return DEFAULT_IMAGE_URL

def fetch_featured_image_from_body(body: str) -> str:
    match = re.search(r"<h2\b[^>]*>(.*?)</h2>", body or "", re.IGNORECASE)
    first_h2 = match.group(1) if match else ""
    return fetch_featured_image(first_h2 or "記事 アイキャッチ", body=body)
