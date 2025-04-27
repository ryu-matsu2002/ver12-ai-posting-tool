# ───────────────────────────────────────────────────────────────
# app/image_utils.py   – v6 (2025-05-XX)
# ───────────────────────────────────────────────────────────────
"""
・Pixabay → Unsplash → デフォルト画像 の順でフォールバック
・検索クエリは呼び出し側で「キーワード＋タイトル＋見出し」などを組み立て
・上位5件からランダム選択し、同一URLの重複利用を避ける
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

# ───── プロセス内で使用済み URL を記憶 ─────
_used_image_urls: set[str] = set()


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


# ══════════════════════════════════════════════
# 重複防止付き Top-5 ランダム選択
# ══════════════════════════════════════════════
def _pick_pixabay(hits: List[dict], keywords: List[str]) -> Optional[str]:
    """
    hits: Pixabay から返ってきたヒット一覧
    keywords: 検索クエリを split() した単語リスト
    ・タグ一致度でスコアリング
    ・上位5件から shuffle → 未使用URLを優先返却
    ・全ヒットから未使用ランダム → 最終的に上位1件
    """
    if not hits:
        return None

    def score(hit: dict) -> int:
        tags = hit.get("tags", "").lower()
        return sum(1 for kw in keywords if kw.lower() in tags)

    # スコア順にソートして上位5件だけ抽出
    top5 = sorted(hits, key=score, reverse=True)[:5]
    random.shuffle(top5)

    # 未使用URLを優先
    for h in top5:
        url = h.get("webformatURL")
        if url and url not in _used_image_urls:
            _used_image_urls.add(url)
            return url

    # フォールバック：全ヒットから未使用をランダム
    unused = [
        h.get("webformatURL")
        for h in hits
        if (u := h.get("webformatURL")) and u not in _used_image_urls
    ]
    if unused:
        choice = random.choice(unused)
        _used_image_urls.add(choice)
        return choice

    # 最終手段：Top5の先頭URL
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
# 公開 API : 記事投稿用アイキャッチ取得
# ══════════════════════════════════════════════
def fetch_featured_image(query: str) -> str:
    """
    1) 呼び出し側で組み立てた query を受け取る
    2) Pixabay でヒット → _pick_pixabay で重複回避しつつ返却
    3) Pixabay 0件 or 排他失敗 → Unsplash Source
    4) それもエラーなら DEFAULT_IMAGE_URL
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
