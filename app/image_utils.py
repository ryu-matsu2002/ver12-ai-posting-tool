# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v8-fixed (2025-05-XX)  *ensure-thumb*
# ───────────────────────────────────────────────────────────────

"""
Pixabay → Unsplash → デフォルト画像の順でアイキャッチを取得するユーティリティ

この改訂版では、画像サイズフィルタを緩和し、
必ず何らかの URL を返すことで WordPress のサムネイル設定を安定化します。
"""

from __future__ import annotations
import os, random, time, logging, requests
from typing import List, Optional

# ───── 設定 ─────
PIXABAY_API_KEY       = os.getenv("PIXABAY_API_KEY", "")
DEFAULT_IMAGE_URL     = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")
PIXABAY_TIMEOUT       = 5
MAX_PER_PAGE          = 30
RECENTLY_USED_TTL     = int(os.getenv("IMAGE_CACHE_TTL", "86400"))  # 24h(@default)

# ビジネス系タグ（＋関連性向上）
_BUSINESS_TAGS = {
    "money", "business", "office", "finance", "analysis", "marketing",
    "startup", "strategy", "computer", "statistics", "success"
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

# ══════════════════════════════════════════════
# スコアリング & 選択
# ══════════════════════════════════════════════
def _score(hit: dict, kw_set: set[str]) -> int:
    tags = {t.strip().lower() for t in hit.get("tags","").split(",")}
    base = sum(1 for kw in kw_set if kw in tags)
    bonus = 2 if tags & _BUSINESS_TAGS else 0
    return base + bonus

def _valid_dim(hit: dict) -> bool:
    # サイズチェックは緩和：640×400 未満でも許容
    w, h = hit.get("imageWidth",0), hit.get("imageHeight",1)
    if w <= 0 or h <= 0:
        return False
    # 縦横比のみチェック
    ratio = w/h
    return 0.5 <= ratio <= 3.0

def _pick_pixabay(hits: List[dict], keywords: List[str]) -> Optional[str]:
    if not hits:
        return None
    kw_set = {k.lower() for k in keywords}
    # スコア上位 10 件
    top = sorted(hits, key=lambda h: _score(h, kw_set), reverse=True)[:10]
    random.shuffle(top)
    # まずは未使用＆サイズOK
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and (not _is_recently_used(url)) and _valid_dim(h):
            _mark_used(url)
            return url
    # 次に未使用のみ
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url):
            _mark_used(url)
            return url
    # 最後にどれか一つ
    url = top[0].get("largeImageURL") or top[0].get("webformatURL")
    if url:
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
    常に URL を返す。Pixabay→Augmented Pixabay→Unsplash→DEFAULT の順。
    """
    try:
        keywords = query.split()
        # 1. 素のクエリ
        hits = _search_pixabay(query)
        url = _pick_pixabay(hits, keywords)
        if url:
            return url

        # 2. ビジネス補強
        aug = query + " business money"
        hits = _search_pixabay(aug)
        url = _pick_pixabay(hits, keywords + ["business","money"])
        if url:
            return url

        # 3. Unsplash
        return _unsplash_src(query)
    except Exception as e:
        logging.error("fetch_featured_image fatal: %s", e)
        return DEFAULT_IMAGE_URL
