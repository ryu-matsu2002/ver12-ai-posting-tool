# app/services/rewrite/providers/openai_search.py
# ------------------------------------------------------------
# OpenAI "検索" プロバイダ
# 目的：
#   - 2系統を提供
#     (A) search   : OpenAIの Search系モデル（gpt-4o(-mini)-search-preview）で直接上位URLを取得
#     (B) rerank   : 既存の候補URL（DDGなど）を OpenAI で再ランキング/ノイズ除去
#   - 失敗時は空リストを返し、上位でのフォールバック（DDGなど）に委ねる
#
# さらに executor 互換のエントリポイント：
#   - search_and_cache_for_article(article_id, limit, lang, gl, force=False)
#     → URL収集 → 必要ならH2/H3軽量抽出 → SerpOutlineCache 保存
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timezone, timedelta

# OpenAI SDK（インストール済み前提: openai>=1.x 系）
try:
    from openai import OpenAI
except Exception:  # SDK が無い環境でも落ちないように
    OpenAI = None  # type: ignore

from app import db
from app.models import Article, SerpOutlineCache

log = logging.getLogger(__name__)

# ===== 環境変数 =====
_MODE = os.environ.get("SERP_OPENAI_MODE", "none").strip().lower()   # "none" | "search"
_DEFAULT_MODEL = os.environ.get("SERP_OPENAI_MODEL", "gpt-4o-mini-search-preview")
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or None   # Azure等で必要なら指定
_RESULT_LIMIT_FALLBACK = 6
_MAX_PER_DOMAIN = int(os.environ.get("SERP_MAX_PER_DOMAIN", "2") or "2")
_CACHE_TTL_DAYS = int(os.environ.get("SERP_CACHE_TTL_DAYS", "14") or "14")
_HEADINGS_FILL_K = int(os.environ.get("SERP_HEADINGS_FILL_K", "3") or "3")

# ===== OpenAI クライアント =====
def _has_openai() -> bool:
    return bool(OpenAI and _API_KEY)

def _mk_client() -> Optional[object]:
    if not _has_openai():
        return None
    try:
        # --- 旧SDK互換: OpenAIクラスがない場合は直接モジュール参照 ---
        import openai
        openai.api_key = _API_KEY
        if _BASE_URL:
            openai.base_url = _BASE_URL
        return openai
    except Exception as e:
        logging.warning(f"[openai_search] _mk_client failed: {e}")
        return None


# ===== 便利関数 =====
def _normalize_url(u: str) -> Optional[str]:
    """http(s)のみ許可、軽量のトラッキング除去。"""
    try:
        if not isinstance(u, str):
            return None
        u = u.strip()
        if not u.startswith(("http://", "https://")):
            return None
        sp = urlsplit(u)
        if sp.query:
            qs = "&".join(
                kv for kv in sp.query.split("&")
                if not kv.lower().startswith(("utm_", "gclid=", "fbclid="))
            )
        else:
            qs = ""
        return urlunsplit((sp.scheme, sp.netloc, sp.path, qs, ""))  # drop fragment
    except Exception:
        return None

def _cap_per_domain(urls: List[str], k: int) -> List[str]:
    out, seen = [], {}
    for u in urls:
        try:
            host = urlsplit(u).netloc.lower()
        except Exception:
            continue
        c = seen.get(host, 0)
        if c >= k:
            continue
        seen[host] = c + 1
        out.append(u)
    return out

def _extract_urls_from_json_object(text: str) -> List[str]:
    """
    Responses API の json_schema（{"urls":[...]}）想定で安全抽出。
    """
    try:
        obj = json.loads((text or "").strip())
        if isinstance(obj, dict) and isinstance(obj.get("urls"), list):
            return [x for x in obj["urls"] if isinstance(x, str)]
    except Exception:
        # 予期せぬ書式でも壊れないように冗長救済
        try:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                obj = json.loads(text[start:end+1])
                if isinstance(obj, dict) and isinstance(obj.get("urls"), list):
                    return [x for x in obj["urls"] if isinstance(x, str)]
        except Exception:
            return []
    return []
 

def _prompt_for_search(keyword: str, limit: int) -> str:
    # モデルには web_search ツールを必ず使わせ、JSONスキーマ準拠の応答だけを要求
    return (
        "You are a web search assistant for Japanese queries. "
        "Use the web_search tool to find high-quality article pages and return ONLY a JSON object "
        'with the shape {"urls": [string, ...]}.\n'
        f"- Up to {limit} result URLs.\n"
        "- Prefer informative article pages with H2/H3 headings\n"
        "- Avoid tag/category index pages and thin/ads pages\n"
        "- Cap: 2 per domain\n"
        "日本語クエリに対して関連性の高い記事URLのみを収集し、JSONオブジェクト（urls配列）だけを返してください。"
    )


def _web_search_urls(keyword: str, limit: int, model: str) -> List[str]:
    """
    OpenAI Chat Completions API（gpt-4o-mini-search-preview）で上位URLを取得。
    このモデルは検索統合型のため、余分なパラメータ（temperature, max_tokens 等）は禁止。
    返却は JSON 形式 {"urls": [...]} として抽出。
    """
    client = _mk_client()
    if not client:
        return []
    try:
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはWeb検索エンジンです。"
                        "次のクエリに関連する日本語のWebページの上位URLを"
                        f"{limit}件以内でJSON形式に出力してください。"
                        "形式: {\"urls\": [\"https://example.com\", ...]}"
                    ),
                },
                {"role": "user", "content": keyword},
            ]
        )

        text = resp.choices[0].message.content if resp.choices else ""
        urls = _extract_urls_from_json_object(text)
        if not urls:
            logging.warning(f"[OPENAI-WEB_SEARCH] no urls extracted for keyword={keyword!r}")
            return []

        urls = [x for x in (_normalize_url(u) for u in urls) if x]
        urls = _cap_per_domain(urls, _MAX_PER_DOMAIN)
        logging.info(f"[OPENAI-WEB_SEARCH] success {len(urls)} urls for '{keyword}'")
        return urls

    except Exception as e:
        logging.warning(f"[OPENAI-WEB_SEARCH] failed: {e}")
        return []

def _search_with_openai(keyword: str, limit: int, model: str) -> List[str]:
    # 直接 web_search を使う一本化
    return _web_search_urls(keyword, limit, model)

# ===== 公開：URL配列のみ返す軽量API =====
def search_top_urls(keyword: str, limit: int = 6, model: str | None = None) -> List[str]:
    """
    - MODE=none   : 空配列（検索なし）
    - MODE=search : OpenAI Responses API の web_search で直接URL取得（フォールバックなし）
    """
    limit = max(1, int(limit or _RESULT_LIMIT_FALLBACK))

    if _MODE not in ("none", "search"):
        return []

    if _MODE == "none":
        return []

    # MODE=search：web_search 一本化（フォールバックなし）
    urls = _search_with_openai(keyword, limit, model or _DEFAULT_MODEL)
    return urls

# ===== executor 互換：URLをキャッシュに保存するエントリポイント =====
def _recent_cache(article_id: int, ttl_days: int) -> Optional[SerpOutlineCache]:
    rec = (
        db.session.query(SerpOutlineCache)
        .filter(SerpOutlineCache.article_id == article_id)
        .order_by(SerpOutlineCache.fetched_at.desc())
        .first()
    )
    if not rec:
        return None
    try:
        age = datetime.now(timezone.utc) - (rec.fetched_at or datetime.now(timezone.utc))
        if age <= timedelta(days=ttl_days):
            return rec
    except Exception:
        return None
    return None

def _fill_headings_if_needed(outlines: List[Dict[str, Any]], k: int, lang: str, gl: str) -> int:
    """
    すべての要素で 'h' が空なら、先頭 K 件だけ軽量見出し抽出を行う。
    返り値: 実際に見出しを埋められた件数
    """
    if not outlines:
        return 0
    if any((o.get("h") or []) for o in outlines):
        return 0
    filled = 0
    try:
        # 遅延 import（循環回避）
        from app.services.rewrite.serp_collector import _fetch_page_outline as fetch_page_outline
    except Exception:
        return 0

    K = max(1, min(k, len(outlines)))
    for o in outlines[:K]:
        url_ = (o or {}).get("url")
        if not url_:
            continue
        try:
            filled_outline = fetch_page_outline(url_, lang=lang, gl=gl)
            if filled_outline and (filled_outline.get("h") or []):
                o["h"] = filled_outline["h"]
                filled += 1
        except Exception:
            continue
    return filled

def search_and_cache_for_article(
    *,
    article_id: int,
    limit: int = 6,
    lang: str = "ja",
    gl: str = "jp",
    force: bool = False,
) -> Dict[str, Any]:
    """
    executor.py から呼ばれる想定の公開関数。
    手順：
      1) 新鮮キャッシュがあれば尊重（force=False時）
      2) 記事の keyword/title を使って URL 収集（MODE=search or rerank。失敗時はDDGフォールバック）
      3) outlines = [{url, title?, snippet?, h?}, ...] を生成
      4) 全要素 h が空なら、先頭K件だけ軽量見出し抽出で h を補完
      5) SerpOutlineCache に保存（JSON列）
    戻り例：
      {'ok': True, 'cache_id': 123, 'saved_count': 6, 'query': 'キーワード'}
      {'ok': True, 'skipped': 'recent_cache', 'cache_id': 456}
      {'ok': False, 'error': '...'}
    """
    try:
        # 0) 記事取得（必要情報だけ）
        art: Optional[Article] = db.session.query(Article).get(article_id)
        if not art:
            return {"ok": False, "error": f"article_not_found:{article_id}"}

        q = (art.keyword or art.title or "").strip()
        if not q:
            return {"ok": False, "error": "empty_query"}

        # 1) 新鮮キャッシュ（forceで無視可）
        if not force:
            fresh = _recent_cache(article_id, _CACHE_TTL_DAYS)
            if fresh:
                return {"ok": True, "skipped": "recent_cache", "cache_id": fresh.id}

        # 2) URL 収集
        urls: List[str] = search_top_urls(q, limit=limit, model=_DEFAULT_MODEL)
        urls = [u for u in urls if isinstance(u, str)]
        outlines: List[Dict[str, Any]] = []

        for u in urls:
            norm = _normalize_url(u)
            if not norm:
                continue
            outlines.append({"url": norm})

        if not outlines:
            return {"ok": False, "error": "no_results"}

        # 3) 全要素 h が空なら、先頭K件だけ軽量見出し抽出
        _fill_headings_if_needed(outlines, _HEADINGS_FILL_K, lang, gl)

        # 4) SerpOutlineCache に保存
        rec = SerpOutlineCache(
            article_id=article_id,
            fetched_at=datetime.now(timezone.utc),
            outlines=outlines,
            query=q,
            # 旧スキーマ互換性のため JSON 内にメタ付与（カラムは無い）
            # "provider": "web_search"
            # "k": len(outlines)
        )
        db.session.add(rec)
        db.session.commit()

        return {
            "ok": True,
            "cache_id": rec.id,
            "saved_count": len(outlines),
            "query": q,
        }
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        log.warning("[openai_search] search_and_cache_for_article failed: %s", e)
        return {"ok": False, "error": str(e)}
