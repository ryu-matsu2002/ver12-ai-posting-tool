# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v8 (2025-05-XX)  *robust + relevance*
# ───────────────────────────────────────────────────────────────
"""
Pixabay → Unsplash → デフォルト画像の順でアイキャッチを取得するユーティリティ

【v8 での主な改良点】
 1. Pixabay を 2 段クエリ:
      ① full query        ② “ビジネス／マネー etc.” 補強語を付与
    → ①が 0 件でも ②で拾える確率が上がる
 2. タグスコアに “ビジネス/学習/マネー” 系タグを +2 ─ 関連性向上
 3. 解像度 640×400 以上・縦横比 0.7〜2.5 を許容 (OGP 想定)
 4. 画像 URL を 24 時間キャッシュ (in-process + redis optional)
 5. 例外が起きても DEFAULT_IMAGE_URL を必ず返す
"""

from __future__ import annotations
import os, random, time, logging, requests
from typing import List, Optional

# ───── 設定 ─────
PIXABAY_API_KEY   = os.getenv("PIXABAY_API_KEY", "")
DEFAULT_IMAGE_URL = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")
PIXABAY_TIMEOUT   = 5
MAX_PER_PAGE      = 30

# ビジネス系タグ
_BUSINESS_TAGS = {
    "money","business","office","finance","analysis","marketing",
    "startup","strategy","computer","statistics","success"
}

# 画像利用履歴 (pid→timestamp) / console 再起動でリセット
_used_image_urls: dict[str, float] = {}

def _is_recently_used(url: str, ttl: int = 86_400) -> bool:
    """同一 URL を 24h 以内に再利用しない"""
    ts = _used_image_urls.get(url)
    return ts is not None and time.time() - ts < ttl

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
        r = requests.get("https://pixabay.com/api/", params=params,
                         timeout=PIXABAY_TIMEOUT)
        if r.status_code == 400:
            logging.warning("Pixabay 400 for %s – fallthrough to Unsplash", query)
            return []                       # ← ここで即 return
        r.raise_for_status()                # ★ 400 以外の例外だけ拾う
        return r.json().get("hits", [])
    except Exception as e:
        logging.debug("Pixabay API error (%s): %s", query, e)
        return []


# ══════════════════════════════════════════════
# スコアリング & 選択
# ══════════════════════════════════════════════
def _score(hit: dict, kw_set: set[str]) -> int:
    tags = {t.strip().lower() for t in hit.get("tags", "").split(",")}
    base = sum(1 for kw in kw_set if kw in tags)
    bonus = 2 if tags & _BUSINESS_TAGS else 0
    return base + bonus

def _valid_dim(hit: dict) -> bool:
    w, h = hit.get("imageWidth", 0), hit.get("imageHeight", 1)
    if w < 640 or h < 400:
        return False
    ratio = w / h
    return 0.7 <= ratio <= 2.5

def _pick_pixabay(hits: List[dict], keywords: List[str]) -> Optional[str]:
    if not hits:
        return None
    kw_set = {k.lower() for k in keywords}
    # スコア上位 10 件をシャッフル
    cand = sorted(hits, key=lambda h: _score(h, kw_set), reverse=True)[:10]
    random.shuffle(cand)
    for h in cand:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if url and not _is_recently_used(url) and _valid_dim(h):
            _mark_used(url)
            return url
    # fallback: 使われていないものが無ければ最初の 1 枚を返す
    url = cand[0].get("webformatURL")
    if url:
        _mark_used(url)
    return url

# ══════════════════════════════════════════════
# Unsplash Source
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    # ── 120字・最大6語に縮める ───────────────────
    words = query.split()[:6]
    short = " ".join(words)[:120]
    q = requests.utils.quote(short)
    return f"https://source.unsplash.com/featured/1200x630/?{q}"
# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════
def fetch_featured_image(query: str) -> str:
    """
    戻り値は常に URL 文字列 (失敗時 DEFAULT_IMAGE_URL)。
    """
    try:
        keywords = query.split()
        # ① そのままのクエリ
        hits = _search_pixabay(query)
        url = _pick_pixabay(hits, keywords)
        if url:
            return url

        # ② ビジネス補強語を足して再試行
        biz_aug = query + " business money"
        hits = _search_pixabay(biz_aug)
        url = _pick_pixabay(hits, keywords + ["business","money"])
        if url:
            return url

        # ③ Unsplash
        return _unsplash_src(query)
    except Exception as e:
        logging.debug("fetch_featured_image fatal: %s", e)
        return DEFAULT_IMAGE_URL
