# app/services/rewrite/providers/openai_search.py
# ------------------------------------------------------------
# OpenAI "検索" プロバイダ
# 目的：
#   - 2系統を提供
#     (A) search   : OpenAIの Search系モデル（gpt-4o(-mini)-search-preview）で直接上位URLを取得
#     (B) rerank   : 既存の候補URL（DDGなど）を OpenAI で再ランキング/ノイズ除去
#   - 失敗時は空リストを返し、上位でのフォールバック（DDGなど）に委ねる/もしくは本モジュール内で暫定フォールバックを実施

# 使い方：
#   .env 例：
#     SERP_PROVIDER=openai
#     SERP_OPENAI_MODE=none | search | rerank   # 既定: none
#     SERP_OPENAI_MODEL=gpt-4o-mini-search-preview
#   MODE=none   : 何もせず空を返す（呼び出し元がDDGへフォールバック）
#   MODE=search : OpenAIのSearchモデルで直接URLを取得（推奨）
#   MODE=rerank : DDG候補の再ランキング
#
# インターフェース：
#   search_top_urls(keyword: str, limit: int = 6, model: str | None = None) -> list[str]
#
# 注意：
#   - ここでは「URLを返すだけ」。本文巡回や見出し抽出は serp_collector 側の責務。
#   - OpenAIキー未設定やAPI失敗時は、静かに「空リスト」を返す（上位でDDGフォールバック）。
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import time
import logging
from typing import List, Dict, Any
from urllib.parse import urlsplit, urlunsplit

# OpenAI SDK（インストール済み前提: openai>=1.x 系）
try:
    from openai import OpenAI
except Exception:  # SDKが無い環境でも落ちないように
    OpenAI = None  # type: ignore

# env 読み込み
_MODE = os.environ.get("SERP_OPENAI_MODE", "none").strip().lower()   # "none" | "search" | "rerank"
_DEFAULT_MODEL = os.environ.get("SERP_OPENAI_MODEL", "gpt-4o-mini-search-preview")
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or None   # Azure等で必要なら指定
_RESULT_LIMIT_FALLBACK = 6
_MAX_PER_DOMAIN = int(os.environ.get("SERP_MAX_PER_DOMAIN", "2") or "2")

# ロガー
log = logging.getLogger(__name__)


def _has_openai() -> bool:
    return bool(OpenAI and _API_KEY)


def _mk_client() -> Any:
    """OpenAIクライアントを生成（BASE_URL対応）。"""
    if not _has_openai():
        return None
    try:
        if _BASE_URL:
            return OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
        return OpenAI(api_key=_API_KEY)
    except Exception:
        return None

def _normalize_url(u: str) -> str | None:
    """http(s)のみ許可、トラッキング最小除去の軽量正規化。"""
    try:
        if not isinstance(u, str):
            return None
        u = u.strip()
        if not u.startswith(("http://", "https://")):
            return None
        sp = urlsplit(u)
        # ごく軽いUTM類の除去（必要十分の最小）
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
    """同一ドメイン上限k件。"""
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
def _prompt_for_rerank(keyword: str, limit: int, items: List[Dict[str, str]]) -> str:
    """
    モデルに渡すプロンプト（system+user）。JSON（URL配列）だけ返すよう強制。
    items: [{"url":..., "title":..., "snippet":...}, ...]
    """
    # 情報漏洩を避けた最小プロンプト：厳格にJSON配列を要求
    return (
        "You are an assistant that re-ranks web results for a given Japanese query.\n"
        "Return ONLY a JSON array of up to {limit} URLs (strings), no extra text.\n"
        "Prefer results that are:\n"
        "- Highly relevant to the query intent\n"
        "- Informational (not top-level homepages or thin pages)\n"
        "- Likely to contain thorough H2/H3 headings and substantive content\n"
        "- Non-duplicate domains, within 2 per domain\n"
        "Demote: category index pages, tag pages, top-level landing pages, ads/spam.\n"
        "日本語クエリに対して関連性が高い順に、最大{limit}件のURLをJSON配列で返してください。"
    ).format(limit=limit)

def _prompt_for_search(keyword: str, limit: int) -> str:
    """
    Searchモデル向けの厳格プロンプト。純粋なURL配列のみを返すことを強制。
    """
    return (
        "You are a Japanese web search assistant. For the given query, return ONLY a JSON array "
        f"of up to {limit} high-quality result URLs (strings). No prose, no markdown, no code block.\n"
        "Guidelines:\n"
        "- Highly relevant to the query intent\n"
        "- Prefer informative article pages with substantial headings and text\n"
        "- Avoid thin pages, tag/category index pages, ads, or pure navigation hubs\n"
        "- Cap: within 2 URLs per domain\n"
        "日本語クエリに対して関連性の高い記事URLを最大{limit}件、JSON配列のみで返してください。"
    ).replace("{limit}", str(limit))

def _rerank_with_openai(keyword: str, limit: int, candidates: List[Dict[str, str]], model: str) -> List[str]:
    """
    DDG候補(candidates)をOpenAIで再ランキングし、URL配列を返す。
    失敗時は空配列（上位フォールバックのため）。
    """
    client = _mk_client()
    if not client:
        return []

    prompt = _prompt_for_rerank(keyword, limit, candidates)
    try:
        # モデルに渡すコンテキストは最小限（タイトル/スニペット/URL）
        # Responses API（SDK 1.x）
        content = {
            "keyword": keyword,
            "limit": limit,
            "candidates": candidates[: max(limit * 4, 12)],  # 少し多めに渡して取捨選択させる
        }
        resp = client.responses.create(
            model=model or _DEFAULT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(content, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            max_output_tokens=256,
        )
        text = (resp.output_text if hasattr(resp, "output_text") else resp.to_dict().get("output_text", "")) or ""
        urls = _extract_json_url_array(text)  # 期待形式： ["https://...","https://..."]
        if urls:
            urls = [x for x in (_normalize_url(u) for u in urls) if x]
            urls = _cap_per_domain(urls, _MAX_PER_DOMAIN)
            return urls[:limit]
        return []
    except Exception as e:
        log.warning("[OPENAI-RERANK] failed: %s", e)
        return []

def _search_with_openai(keyword: str, limit: int, model: str) -> List[str]:
    """
    OpenAIの Search系モデル（chat.completions/Responses互換）で直接URL配列を取得。
    失敗時は空配列（呼び出し元フォールバック）。
    """
    client = _mk_client()
    if not client:
        return []
    sys_prompt = _prompt_for_search(keyword, limit)
    try:
        # Chat Completions経由での呼び出し（SDK 1.x）
        resp = client.chat.completions.create(
            model=model or _DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": keyword},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        urls = _extract_json_url_array(text)
        if urls:
            urls = [x for x in (_normalize_url(u) for u in urls) if x]
            urls = _cap_per_domain(urls, _MAX_PER_DOMAIN)
            return urls[:limit]
        return []
    except Exception as e:
        log.warning("[OPENAI-SEARCH] failed: %s", e)
        return []


def _extract_json_url_array(text: str) -> List[str]:
    """
    モデル出力から最初のJSON配列を見つけてURL配列にする。
    """
    text = (text or "").strip()
    # そのままJSON配列で来た場合
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str)]
        except Exception:
            pass

    # コードブロックなどに包まれている場合の救済
    # ```json\n[ ... ]\n```
    try:
        start = text.find("[")
        end = text.rfind("]")
        if 0 <= start < end:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str)]
    except Exception:
        pass
    return []


def _ddg_candidates(keyword: str, limit: int) -> List[Dict[str, str]]:
    """
    serp_collector の DDG 取得関数を使って暫定候補を得る。
    import ループを避けるため、遅延インポート。
    """
    try:
        from app.services.rewrite.serp_collector import _search_top_urls_duckduckgo
        # DDGは title/snippet 付きの dict を返す
        return _search_top_urls_duckduckgo(keyword, limit=limit, lang="ja", gl="jp")
    except Exception:
        return []


def search_top_urls(keyword: str, limit: int = 6, model: str | None = None) -> List[str]:
    """
    公開インターフェース：
      - MODE=none   : 何もせず空を返す（呼び出し元がDDGフォールバック）
      - MODE=rerank : DDG候補→OpenAIで再ランキング→URL配列
    """
    limit = max(1, int(limit or _RESULT_LIMIT_FALLBACK))

    if _MODE not in ("none", "search", "rerank"):
        # 未知値は安全側で none 扱い
        return []

    if _MODE == "none":
        return []

    if _MODE == "search":
        # 直接検索（OpenAI）。失敗時はDDGへフォールバック。
        urls = _search_with_openai(keyword, limit, model or _DEFAULT_MODEL)
        if urls:
            return urls
        # 準フォールバック：DDG上位
        cands = _ddg_candidates(keyword, max(limit * 2, 10))
        return [c["url"] for c in cands[:limit] if isinstance(c, dict) and c.get("url")]

    # rerank モード
    cands = _ddg_candidates(keyword, max(limit * 3, 12))
    if not cands:
        return []
    if not _has_openai():
        return [c["url"] for c in cands[:limit] if isinstance(c, dict) and c.get("url")]
    urls = _rerank_with_openai(keyword, limit, cands, model or _DEFAULT_MODEL)
    if urls:
        return urls
    return [c["url"] for c in cands[:limit] if isinstance(c, dict) and c.get("url")]
