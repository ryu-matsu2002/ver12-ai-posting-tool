# ─────────────────────────────────────────────
# app/image_utils.py   – v3 (2025-04-XX)
# ─────────────────────────────────────────────
"""
・Pixabay で 0 件なら Unsplash Source をフォールバック
・tags にキーワードを含む画像を優先（含まなければ最初のヒット）
・全て標準ライブラリ／requests のみで完結（追加 pip 不要）
"""

from __future__ import annotations
import os, requests, random, logging
from typing import List, Optional

# ───── API キー ─────
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

# ══════════════════════════════════════════════
# Pixabay
# ══════════════════════════════════════════════
def _search_pixabay(query: str, per_page: int = 20) -> List[dict]:
    if not PIXABAY_API_KEY:
        return []
    params = {
        "key"        : PIXABAY_API_KEY,
        "q"          : query,
        "image_type" : "photo",
        "per_page"   : per_page,
        "safesearch" : "true",
    }
    try:
        r = requests.get("https://pixabay.com/api/", params=params, timeout=6)
        r.raise_for_status()
        return r.json().get("hits", [])
    except Exception as e:
        logging.debug("Pixabay API エラー (%s): %s", query, e)
        return []

def _pick_pixabay(hits: List[dict], keyword: str = "") -> Optional[str]:
    """
    tags に keyword を含み 0.5 < w/h < 3 の画像を返す。
    見つからなければ最初のヒットを返す。
    """
    if not hits:
        return None
    random.shuffle(hits)
    for h in hits:
        tags = h.get("tags", "").lower()
        if keyword and keyword.lower() not in tags:
            continue
        w, hgt = h.get("imageWidth", 0), h.get("imageHeight", 1)
        if hgt and 0.5 < w / hgt < 3:
            return h.get("webformatURL")
    # 条件に合う物が無ければ 1 件目
    return hits[0].get("webformatURL")

# ══════════════════════════════════════════════
# Unsplash (フォールバック)
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    """
    Unsplash Source は API キー不要・ランダム返却。
    例: https://source.unsplash.com/featured/1200x630/?travel
    """
    q = requests.utils.quote(query or "travel")
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

# ══════════════════════════════════════════════
# Public
# ══════════════════════════════════════════════
def fetch_featured_image(body: str, keyword: str = "") -> str:
    """
    ① 本文 → 先頭 50 文字をとりあえず検索語として Pixabay
    ② tags に一致する画像を優先 / 無ければ最初
    ③ 0 件なら Unsplash Source でフォールバック
    """
    # ------ ① 検索語決定 ------
    # 深い NLP を使わず「本文先頭 50 字 or keyword」で十分ヒットする
    query = (keyword or body[:50]).strip()
    if not query:
        return ""

    # ------ ② Pixabay ------
    hits = _search_pixabay(query)
    url = _pick_pixabay(hits, keyword=query)
    if url:
        return url

    # ------ ③ Unsplash ------
    return _unsplash_src(query)
