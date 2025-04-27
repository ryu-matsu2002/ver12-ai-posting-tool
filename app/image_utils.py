# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v7 (2025-05-XX)   *diversified*
# ───────────────────────────────────────────────────────────────
"""
Pixabay → Unsplash → デフォルト画像の順でアイキャッチを取得するユーティリティ

■ 改良ポイント（v6 → v7）
  1. Pixabay API を 1 回 30 件取得し母数を拡大
  2. タグ一致スコアの **上位 5 件**を `random.sample` で完全シャッフル  
     └ 順序バイアスを排除して毎回画像がバラける
  3. **縦横比 0.5–3** 以外は除外して極端な縦長横長を避ける
  4. 同一プロセス内で選択済み URL を `_used_image_urls` セットに保持し  
     画像の **重複利用を防止**
"""

from __future__ import annotations
import os, random, requests, logging
from typing import List, Optional

# ───── 環境変数 ─────
PIXABAY_API_KEY   = os.getenv("PIXABAY_API_KEY", "")
DEFAULT_IMAGE_URL = os.getenv("DEFAULT_IMAGE_URL", "/static/default-thumb.jpg")

# ───── プロセス内で使用済み URL を記憶 ─────
_used_image_urls: set[str] = set()

# ══════════════════════════════════════════════
# Pixabay 検索
# ══════════════════════════════════════════════
def _search_pixabay(query: str, per_page: int = 30) -> List[dict]:
    """query が空、または APIキー未設定なら空配列を返す"""
    if not PIXABAY_API_KEY or not query:
        return []
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": per_page,
        "safesearch": "true",
    }
    try:
        r = requests.get("https://pixabay.com/api/", params=params, timeout=6)
        r.raise_for_status()
        return r.json().get("hits", [])
    except Exception as e:
        logging.debug("Pixabay API error (%s): %s", query, e)
        return []

# ══════════════════════════════════════════════
# 重複防止付き Top-5 ランダム選択
# ══════════════════════════════════════════════
def _pick_pixabay(hits: List[dict], keywords: List[str]) -> Optional[str]:
    """
    ・タグ一致度でスコアリング
    ・スコア上位 5 件をランダム順に試行
    ・縦横比 0.5〜3 かつ未使用 URL を優先
    """
    if not hits:
        return None

    def score(hit: dict) -> int:
        tags = hit.get("tags", "").lower()
        return sum(1 for kw in keywords if kw.lower() in tags)

    top5 = sorted(hits, key=score, reverse=True)[:5]
    for h in random.sample(top5, k=len(top5)):
        url = h.get("webformatURL")
        w, hgt = h.get("imageWidth", 0), h.get("imageHeight", 1)
        if url and url not in _used_image_urls and hgt and 0.5 < w / hgt < 3:
            _used_image_urls.add(url)
            return url

    # 上位 5 件で見つからなければ、全ヒットから未使用をランダム
    random.shuffle(hits)
    for h in hits:
        url = h.get("webformatURL")
        w, hgt = h.get("imageWidth", 0), h.get("imageHeight", 1)
        if url and url not in _used_image_urls and hgt and 0.5 < w / hgt < 3:
            _used_image_urls.add(url)
            return url

    # すべて使用済みの場合は top5 先頭を許容
    fallback = top5[0].get("webformatURL")
    if fallback:
        _used_image_urls.add(fallback)
    return fallback

# ══════════════════════════════════════════════
# Unsplash Source（Pixabayフォールバック）
# ══════════════════════════════════════════════
def _unsplash_src(query: str) -> str:
    q = requests.utils.quote(query or "")
    return f"https://source.unsplash.com/featured/1200x630/?{q}"

# ══════════════════════════════════════════════
# Public API : 記事投稿用アイキャッチ取得
# ══════════════════════════════════════════════
def fetch_featured_image(query: str) -> str:
    """
    1) 呼び出し側で組み立てた query を受け取る
    2) Pixabay ヒット → _pick_pixabay で重複回避しつつ返却
    3) Pixabay 0件 / 失敗 → Unsplash Source
    4) さらに失敗 → DEFAULT_IMAGE_URL
    """
    keywords = query.split()
    try:
        hits = _search_pixabay(query)
        url  = _pick_pixabay(hits, keywords)
        if url:
            return url
    except Exception:
        logging.debug("Pixabay fallback failed for query: %s", query)

    try:
        return _unsplash_src(query)
    except Exception:
        logging.debug("Unsplash fallback failed for query: %s", query)

    return DEFAULT_IMAGE_URL
