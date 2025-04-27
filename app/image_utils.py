# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v5 (2025-05-XX)
# ───────────────────────────────────────────────────────────────
"""
・Pixabay → Unsplash → デフォルト画像 の順でフォールバック
・検索クエリは呼び出し側で「キーワード＋タイトル＋見出し」などを組み立て
・エラー時や全て 0 件なら社内 CDN 等の DEFAULT_IMAGE_URL を返却
"""

from __future__ import annotations
import os
import random
import requests
import logging
from typing import List, Optional

# ───── 環境変数 ─────
PIXABAY_API_KEY   = os.getenv("PIXABAY_API_KEY", "")
DEFAULT_IMAGE_URL = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")

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

def _pick_pixabay(hits: List[dict], keywords: List[str]) -> Optional[str]:
    """
    hits: Pixabay から返ってきたヒット一覧
    keywords: 検索クエリを split() した単語リスト
    タグ中により多くマッチする画像ほど優先的に返す
    """
    if not hits:
        return None

    def score(hit: dict) -> int:
        tags = hit.get("tags", "").lower()
        # 各キーワードの出現回数をスコア化
        return sum(1 for kw in keywords if kw.lower() in tags)

    # スコア順にソートし、上位5件だけをランダム
    top_n = sorted(hits, key=score, reverse=True)[:5]
    random.shuffle(top_n)
    for h in top_n:
        w, hgt = h.get("imageWidth", 0), h.get("imageHeight", 1)
        if hgt and 0.5 < w / hgt < 3:
            return h.get("webformatURL")

    # 上位5件がすべてNGなら、元のヒットからランダムで返却
    fallback = random.choice(hits)
    return fallback.get("webformatURL")

# ══════════════════════════════════════════════
# Unsplash Source（Pixabayフォールバック）
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    q = requests.utils.quote(query or "")
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════
def fetch_featured_image(query: str) -> str:
    """
    1) 呼び出し側で組み立てた query を受け取る
    2) Pixabay でヒット → タグ一致度と縦横比で最適なものを返却
    3) Pixabay 0件 or エラー → Unsplash Source
    4) それもエラーなら DEFAULT_IMAGE_URL
    """
    # --- Pixabay検索 ---
    try:
        hits = _search_pixabay(query)
        # query をスペース区切りでキーワードリスト化
        keywords = query.split()
        url = _pick_pixabay(hits, keywords)
        if url:
            return url
    except Exception:
        logging.debug("Pixabay fallback failed for query: %s", query)

    # --- Unsplash フォールバック ---
    try:
        return _unsplash_src(query)
    except Exception:
        logging.debug("Unsplash fallback failed for query: %s", query)

    # --- 最終フォールバック ---
    return DEFAULT_IMAGE_URL
