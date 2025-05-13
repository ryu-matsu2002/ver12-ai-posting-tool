# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v8-fixed2 (2025-05-XX)  *always-thumb*
# ───────────────────────────────────────────────────────────────

"""
Pixabay → Unsplash → デフォルト画像の順でアイキャッチを取得するユーティリティ

この改訂版では fetch_featured_image が絶対に文字列を返し、
None を返さないことでプレビュー／WP投稿時に “thumb” のまま
表示されなくなる問題を解消します。
"""

from __future__ import annotations
import os, random, time, logging, requests
from typing import List
from flask import current_app

# ───── 設定 ─────
ROOT_URL            = os.getenv("APP_ROOT_URL", "https://your-domain.com")
PIXABAY_API_KEY     = os.getenv("PIXABAY_API_KEY", "")
PIXABAY_TIMEOUT     = 5
MAX_PER_PAGE        = 30
RECENTLY_USED_TTL   = int(os.getenv("IMAGE_CACHE_TTL", "86400"))  # 24h
DEFAULT_IMAGE_PATH  = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")
# プレビュー表示／WP 投稿時に使う絶対 URL
DEFAULT_IMAGE_URL   = (
    DEFAULT_IMAGE_PATH.startswith("http")
    and DEFAULT_IMAGE_PATH
    or f"{ROOT_URL}{DEFAULT_IMAGE_PATH}"
)

# ビジネス系タグ（関連性向上）
_BUSINESS_TAGS = {
    "money","business","office","finance","analysis",
    "marketing","startup","strategy","computer",
    "statistics","success"
}

# 画像利用履歴 (url→timestamp)
_used_image_urls: dict[str, float] = {}

def _is_recently_used(url: str, ttl: int = RECENTLY_USED_TTL) -> bool:
    ts = _used_image_urls.get(url)
    return ts is not None and (time.time() - ts) < ttl

def _mark_used(url: str) -> None:
    _used_image_urls[url] = time.time()

# ══════════════════════════════════════════════
# Pixabay 検索
# ══════════════════════════════════════════════
def _search_pixabay(query: str, per_page: int = MAX_PER_PAGE) -> List[dict]:
    if not PIXABAY_API_KEY or not query:
        return []
    query = query.replace("　"," ").strip()
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

# ══════════════════════════════════════════════
# スコアリング & 選択
# ══════════════════════════════════════════════
def _score(hit: dict, kw_set: set[str]) -> int:
    tags  = {t.strip().lower() for t in hit.get("tags","").split(",")}
    base  = sum(1 for kw in kw_set if kw in tags)
    bonus = 2 if tags & _BUSINESS_TAGS else 0
    return base + bonus

def _valid_dim(hit: dict) -> bool:
    w, h = hit.get("imageWidth",0), hit.get("imageHeight",1)
    if w<=0 or h<=0:
        return False
    ratio = w/h
    return 0.5 <= ratio <= 3.0

def _pick_pixabay(hits: List[dict], keywords: List[str]) -> str:
    # Optional[str] → str に変更。必ず最後は文字列を返す
    if not hits:
        return DEFAULT_IMAGE_URL
    kw_set = {k.lower() for k in keywords}
    top = sorted(hits, key=lambda h: _score(h, kw_set), reverse=True)[:10]
    random.shuffle(top)
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url) and _valid_dim(h):
            _mark_used(url)
            return url
    # フォールバック順序
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url):
            _mark_used(url)
            return url
    url = top[0].get("largeImageURL") or top[0].get("webformatURL") or DEFAULT_IMAGE_URL
    _mark_used(url)
    return url

# ══════════════════════════════════════════════
# Unsplash Source
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    words = query.split()[:6]
    short = " ".join(words)[:120]
    q = requests.utils.quote(short)
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════
def fetch_featured_image(query: str) -> str:
    """
    絶対に文字列を返す。
    Pixabay→Augmented Pixabay→Unsplash→DEFAULT_URL の順。
    """
    def is_safe_image(url: str) -> bool:
        return url.lower().endswith(('.jpg', '.jpeg', '.png'))

    try:
        keywords = query.split()

        # 1. 素のクエリ
        hits = _search_pixabay(query)
        url  = _pick_pixabay(hits, keywords)
        if url and is_safe_image(url):
            return url

        # 2. ビジネス補強
        hits = _search_pixabay(query + " business money")
        url  = _pick_pixabay(hits, keywords + ["business","money"])
        if url and is_safe_image(url):
            return url

        # 3. Unsplash
        unsplash_url = _unsplash_src(query)
        if is_safe_image(unsplash_url):
            return unsplash_url

        # fallback
        return DEFAULT_IMAGE_URL
    except Exception as e:
        logging.error("fetch_featured_image fatal: %s", e)
        return DEFAULT_IMAGE_URL

