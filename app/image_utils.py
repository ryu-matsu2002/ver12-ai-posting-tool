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
import re
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

# ✅【追加】URLが画像形式かどうかをHEADで確認
def _is_image_url(url: str) -> bool:
    try:
        r = requests.head(url, timeout=5)
        content_type = r.headers.get("Content-Type", "")
        return content_type.startswith("image/")
    except Exception as e:
        logging.warning(f"[画像判定失敗] {url} → {e}")
        return False    

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

# ✅【修正】画像形式かどうかのチェックを追加
def _pick_pixabay(hits: List[dict], keywords: List[str]) -> str:
    if not hits:
        return DEFAULT_IMAGE_URL
    kw_set = {k.lower() for k in keywords}
    top = sorted(hits, key=lambda h: _score(h, kw_set), reverse=True)[:10]
    random.shuffle(top)

    # 🔍 最優先で画像形式・未使用・サイズ適正なURLを返す
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url) and _valid_dim(h) and _is_image_url(url):
            _mark_used(url)
            return url

    # 🔁 サイズ不問でも画像形式であればOK
    for h in top:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url) and _is_image_url(url):
            _mark_used(url)
            return url

    # 🔚 最後のフォールバック（確認なし）
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
    高精度なアイキャッチ画像を取得する。
    クエリから不要語除去・英語化・補強語追加・Pixabay→Unsplash→デフォルトの順で対応。
    """
    def is_safe_image(url: str) -> bool:
        return url.lower().endswith(('.jpg', '.jpeg', '.png'))

    def clean_and_translate(query: str) -> str:
        stopwords = {"の", "に", "で", "を", "と", "が", "は", "から", "ため", "こと", "について", "方法", "おすすめ", "完全ガイド"}
        jp_to_en = {
            "ピラティス": "pilates", "腰痛": "back pain", "改善": "improvement", "福岡": "fukuoka",
            "英語": "english", "副業": "side job", "ブログ": "blog", "ビジネス": "business",
            "旅行": "travel", "脱毛": "hair removal", "メイク": "makeup", "占い": "fortune telling"
        }
        keywords = [w for w in re.split(r"[\\s　]+", query) if w and w not in stopwords]
        translated = [jp_to_en.get(w, w) for w in keywords]
        return " ".join(translated)

    try:
        base_query = clean_and_translate(query)
        keywords = base_query.split()

        # 1. 素のクエリ
        hits = _search_pixabay(base_query)
        url  = _pick_pixabay(hits, keywords)
        if url and is_safe_image(url):
            return url

        # 2. 補強検索（補助語追加）
        enhanced_query = f"{base_query} woman person lifestyle"
        hits = _search_pixabay(enhanced_query)
        url = _pick_pixabay(hits, keywords + ["woman", "person", "lifestyle"])
        if url and is_safe_image(url):
            return url

        # 3. Unsplash fallback
        unsplash_url = _unsplash_src(base_query)
        if is_safe_image(unsplash_url):
            return unsplash_url

        return DEFAULT_IMAGE_URL

    except Exception as e:
        logging.error("fetch_featured_image fatal: %s", e)
        return DEFAULT_IMAGE_URL


def fetch_featured_image_from_body(body_html: str, keyword: str) -> str:
    from bs4 import BeautifulSoup

    def extract_top_topics(html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        # h2とh3をすべて取得し、見出し語句の上位3件を選定
        headings = soup.find_all(["h2", "h3"])
        phrases = [h.get_text().strip() for h in headings]
        return phrases[:3] if phrases else [keyword]

    try:
        topics = extract_top_topics(body_html)
        queries = [f"{keyword} {t}" for t in topics]

        for query in queries:
            hits = _search_pixabay(query)
            url = _pick_pixabay(hits, query.split())
            if url and url.lower().endswith((".jpg", ".jpeg", ".png")):
                return url

        # 最後に Unsplash fallback
        return _unsplash_src(f"{keyword} {topics[0]}")
    except Exception as e:
        logging.error(f"画像選定失敗: {e}")
        return DEFAULT_IMAGE_URL
