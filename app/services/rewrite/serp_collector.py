# app/services/rewrite/serp_collector.py
# ------------------------------------------------------------
# SERP収集器（実働版）
# - キーワードでGoogle検索
# - 上位URLを取得（既定6件）
# - 各ページから H2/H3 見出しを軽量抽出
# - SerpOutlineCache に保存（履歴として新規行を追加）
#
# 高速・省メモリの工夫
# - Playwrightの起動は1回だけ（検索→巡回まで同一コンテキスト）
# - 画像/フォント/CSS/メディア/追跡系をブロック
# - タイムアウト短め、件数も控えめ
# - DOMから直接抽出（正規表現はフォールバック用）
# ------------------------------------------------------------

from __future__ import annotations

import re
import time
from typing import List, Dict, Optional
from urllib.parse import quote

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from app import db
from app.models import Article, SerpOutlineCache


# ---------- 軽量テキストユーティリティ ----------

_H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", flags=re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", flags=re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_tags(s: str) -> str:
    if not s:
        return ""
    s = _TAG_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _unique_keep_order(items: List[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return out


# ---------- Playwright セッション（ブロッキング軽減設定） ----------

class _PWSession:
    """起動オーバーヘッドを最小化しつつ、帯域消費を抑える設定で使うための薄いラッパ。"""

    def __init__(self, lang: str = "ja", gl: str = "jp") -> None:
        self._pw = None
        self._browser = None
        self._ctx = None
        self._lang = lang
        self._gl = gl

    def __enter__(self) -> "_PWSession":
        self._pw = sync_playwright().start()
        # --no-sandbox は多くのサーバ環境で必要
        self._browser = self._pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self._ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            locale=f"{self._lang}-{self._gl.upper()}",
        )
        # 重いリソースは捨てる
        def _route_interceptor(route):
            req = route.request
            rtype = req.resource_type
            url = req.url
            if rtype in {"image", "font", "media", "stylesheet"}:
                return route.abort()
            # 追跡/広告/計測系をざっくりブロック（誤爆を避け簡素化）
            if any(bad in url for bad in ("/ads?", "doubleclick.net", "googletag", "analytics", "facebook.com/tr")):
                return route.abort()
            return route.continue_()
        self._ctx.route("**/*", _route_interceptor)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._ctx:
                self._ctx.close()
        finally:
            try:
                if self._browser:
                    self._browser.close()
            finally:
                if self._pw:
                    self._pw.stop()

    def new_page(self):
        return self._ctx.new_page()


# ---------- Google 検索（上位URLだけ取る・軽量） ----------

def _search_top_urls(keyword: str, *, limit: int = 6, lang: str = "ja", gl: str = "jp",
                     timeout_ms: int = 18000) -> List[str]:
    """Googleでキーワード検索し、検索結果の 'h3 を持つリンク' から外部URLを抽出。"""
    if limit <= 0:
        return []
    q = quote(keyword)
    # num= は 10までしか効かないが、上限をかけておく
    url = f"https://www.google.com/search?q={q}&hl={lang}&gl={gl}&num={min(limit, 10)}"

    urls: List[str] = []
    with _PWSession(lang=lang, gl=gl) as sess:
        page = sess.new_page()
        try:
            page.goto(url, timeout=timeout_ms)
            # h3要素をヘッダとして持つアンカー（一般的なオーガニック結果）
            page.wait_for_selector("a h3", timeout=timeout_ms // 2)
            # Playwright の locator は軽い
            # a:has(h3) → 直近の a を取れる（google 内部リンクは除外）
            anchors = page.locator("a:has(h3)")
            count = min(anchors.count(), limit * 2)  # 余裕をみて取り、あとでフィルタ
            for i in range(count):
                try:
                    href = anchors.nth(i).get_attribute("href")
                except Exception:
                    href = None
                if not href:
                    continue
                # Google内部は除外
                if "google." in href:
                    continue
                urls.append(href)
                if len(urls) >= limit:
                    break
        except PWTimeout:
            # 部分結果でよい。空のままでもOK。
            pass
    return _unique_keep_order(urls, limit)


# ---------- ページから見出し抽出（高速・省メモリ） ----------

def _extract_headings_from_dom(page, *, timeout_ms: int = 16000) -> List[str]:
    """DOMから h2,h3 の innerText を直接抜く（最速）。失敗時は [] を返す。"""
    try:
        # 早めに何か来たらOKにする
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        # ネットワーク待ちは最小限
        time.sleep(0.15)
        texts = page.locator("h2, h3").all_text_contents()
        # 余計な空白を軽く整形
        cleaned = [ _WS_RE.sub(" ", (t or "")).strip() for t in texts ]
        # 似た見出しの連打を削る
        return _unique_keep_order([t for t in cleaned if len(t) >= 2], 60)
    except Exception:
        return []


def _extract_headings_from_html(html: str) -> List[str]:
    """フォールバック：HTML文字列からH2/H3を正規表現で抽出。"""
    h2s = [_strip_tags(m.group(1)) for m in _H2_RE.finditer(html or "")]
    h3s = [_strip_tags(m.group(1)) for m in _H3_RE.finditer(html or "")]
    return _unique_keep_order([*h2s, *h3s], 60)


def _fetch_page_outline(url: str, *, timeout_ms: int = 18000, lang: str = "ja", gl: str = "jp") -> Dict:
    """
    URLを開いて H2/H3 の配列を返す。DOM抽出がダメでも HTML からフォールバック。
    失敗しても {"url": url, "h": []} を返して落ちない。
    """
    with _PWSession(lang=lang, gl=gl) as sess:
        page = sess.new_page()
        try:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            # DOM優先（高速）。空ならHTMLから。
            heads = _extract_headings_from_dom(page, timeout_ms=max(8000, timeout_ms // 2))
            if not heads:
                html = page.content()
                heads = _extract_headings_from_html(html or "")
            return {"url": url, "h": heads}
        except Exception:
            return {"url": url, "h": []}


# ---------- 収集・保存パイプライン ----------

def collect_serp_outlines_for_keyword(keyword: str, *, limit: int = 6,
                                      lang: str = "ja", gl: str = "jp") -> List[Dict]:
    """
    キーワードでGoogle検索 → 上位URLを巡回 → 各ページのH2/H3を抽出して返却。
    戻り値: [{url, h:[...]}, ...]
    """
    urls = _search_top_urls(keyword, limit=limit, lang=lang, gl=gl)
    outlines: List[Dict] = []
    # 1URLずつ短時間で取りにいく（並列はメモリ/帯域コストが跳ねるので避ける）
    for u in urls:
        outlines.append(_fetch_page_outline(u, lang=lang, gl=gl))
    return outlines


def cache_outlines(article_id: int, outlines: List[Dict]) -> Dict:
    """
    SerpOutlineCacheに保存（新規行として追加）。
    既存キャッシュを消さず、最新を参照したい場合は executed_at/fetched_at の新しいレコードを使う想定。
    """
    rec = SerpOutlineCache(article_id=article_id, outlines=outlines)
    db.session.add(rec)
    db.session.commit()
    return {"article_id": article_id, "saved_count": len(outlines), "cache_id": rec.id}


def collect_and_cache_for_article(article_id: int, *, limit: int = 6,
                                  lang: str = "ja", gl: str = "jp") -> Dict:
    """
    Article.id から keyword（なければ title）を使ってSERP収集→キャッシュへ保存。
    """
    art = db.session.get(Article, article_id)
    if not art:
        return {"ok": False, "error": f"article_id={article_id} not found"}

    query = (art.keyword or art.title or "").strip()
    if not query:
        return {"ok": False, "error": "no keyword or title to search"}

    outlines = collect_serp_outlines_for_keyword(query, limit=limit, lang=lang, gl=gl)
    saved = cache_outlines(article_id, outlines)
    return {"ok": True, "query": query, **saved}
