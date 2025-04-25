# ─────────────────────────────────────────────
# app/image_utils.py   – v4 (2025-05-XX)
# ─────────────────────────────────────────────
"""
・Pixabay → Unsplash → デフォルト画像 の順でフォールバック
・検索クエリを「キーワード＋本文先頭 H2 見出し」など複合化
・エラー時や全て 0 件なら社内 CDN 等のDEFAULT_IMAGE_URLを返却
"""

from __future__ import annotations
import os, re, requests, random, logging
from typing import List, Optional

# ───── 環境変数 ─────
PIXABAY_API_KEY    = os.getenv("PIXABAY_API_KEY", "")
DEFAULT_IMAGE_URL  = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")

# ══════════════════════════════════════════════
# Pixabay 検索
# ══════════════════════════════════════════════
def _search_pixabay(query: str, per_page: int = 20) -> List[dict]:
    if not PIXABAY_API_KEY or not query:
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
        logging.debug("Pixabay API error (%s): %s", query, e)
        return []

def _pick_pixabay(hits: List[dict], keyword: str = "") -> Optional[str]:
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
    # 条件に合うものが無ければ最初のヒット
    return hits[0].get("webformatURL")

# ══════════════════════════════════════════════
# Unsplash Source（Pixabayフォールバック）
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    q = requests.utils.quote(query or "")
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════
def fetch_featured_image(body: str, keyword: str = "") -> str:
    """
    1) 検索語を「keyword + 本文先頭 H2 見出し」から組み立て
    2) Pixabay でヒット→条件に合うものを返却
    3) Pixabay 0件 or エラー→Unsplash Source
    4) それもエラーなら DEFAULT_IMAGE_URL
    """
    # --- クエリ組み立て ---
    # 本文中の最初の <h2> を抽出
    heading_match = re.search(r"<h2\b[^>]*>(.*?)</h2>", body or "", re.IGNORECASE)
    parts = []
    if keyword:
        parts.append(keyword.strip())
    if heading_match:
        parts.append(heading_match.group(1).strip())
    query = " ".join(parts).strip() or (body or "")[:50].strip()

    # --- Pixabay検索 ---
    try:
        hits = _search_pixabay(query)
        url = _pick_pixabay(hits, keyword=query)
        if url:
            return url
    except Exception:
        logging.debug("Pixabay fallback failed for query: %s", query)

    # --- Unsplashフォールバック ---
    try:
        return _unsplash_src(query)
    except Exception:
        logging.debug("Unsplash fallback failed for query: %s", query)

    # --- 最終フォールバック ---
    return DEFAULT_IMAGE_URL
