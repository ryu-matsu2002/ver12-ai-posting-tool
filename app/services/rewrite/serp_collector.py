# app/services/rewrite/serp_collector.py
# ------------------------------------------------------------
# SERP収集器（実働版）
# （堅牢化版：リンク抽出フェイルオーバー／同意検知／0件時ワンリトライ）
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
import os
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote, urlparse, parse_qs, unquote

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright.sync_api import Error as PWError

from app import db
from app.models import Article, SerpOutlineCache


# ---------- 軽量テキストユーティリティ ----------

_H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", flags=re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", flags=re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# --- デバッグ出力先（0件時の証拠保存） ---
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DBG_DIR = os.path.join(_BASE_DIR, "runtime", "serp_debug")
os.makedirs(_DBG_DIR, exist_ok=True)
_NOW = lambda: time.strftime("%Y%m%d-%H%M%S")


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
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/141.0.0.0 Safari/537.36"
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
        # Google SERPは軽く、外部ページ巡回も同じ方針で帯域節約
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


# ---------- Google SERPの判定・抽出ヘルパ ----------

def _is_consent_or_block_page(title_text: str, body_text: str) -> bool:
    """
    Google の同意/CAPTCHA/一時ブロック画面をざっくり検知。
    """
    t = (title_text or "").lower()
    b = (body_text or "").lower()
    hints = [
        "before you continue to google",      # EN consent
        "確認のためにお手伝いください",             # JA captcha-ish
        "一時的にアクセスできません",               # JP temp block
        "unusual traffic",                    # EN rate limit
        "to continue, please verify",         # generic
    ]
    return any(h in t or h in b for h in hints)

def _normalize_google_redirect(href: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Googleの /url リダイレクトを外部の実URLに展開する。
    戻り値: (実URL or None, 展開を行ったかどうか)
    """
    if not href:
        return None, False
    try:
        u = urlparse(href)
        # 典型: https://www.google.com/url?q=https://example.com/....&sa=...
        if u.netloc.endswith("google.com") and u.path == "/url":
            q = parse_qs(u.query).get("q", [])
            if q:
                # q は URL エンコードされていることが多い
                real = unquote(q[0])
                # 明らかに外部URLだけ採用
                if real.startswith("http://") or real.startswith("https://"):
                    return real, True
    except Exception:
        pass
    return href, False


def _is_google_internal(href: Optional[str]) -> bool:
    """
    Google内部/特殊枠を除外（news/maps/shopping/images 等）。
    なお、/url?q=... は _normalize_google_redirect 側で外部URLへ展開するため、
    ここでは「純粋に内部だけ」を弾く判定にする。
    """
    if not href:
        return True
    href_l = href.lower()
    # /url?q=... は別途展開するので、ここでは即除外しない
    if href_l.startswith("http://google.") or href_l.startswith("https://google."):
        # ただし /url 以外（=純内部）は除外
        try:
            u = urlparse(href)
            if u.path != "/url":
                return True
        except Exception:
            return True
    # 特定サービスの痕跡（念のため二重で）
    bad_starts = (
        "/search?", "/imgres?", "/maps", "/news", "/gws/", "/shopping", "/aclk",
    )
    return href_l.startswith(("http://google.", "https://google.", *bad_starts))

def _normalize_and_filter_href(href: Optional[str]) -> Optional[str]:
    """
    href を正規化（/url?q=... 展開）し、Google内部リンクなら None。
    """
    if not href:
        return None
    # /url?q=... を実URLに展開
    real, expanded = _normalize_google_redirect(href)
    if not real:
        return None
    return None if _is_google_internal(real) else real


def _extract_result_links(page, *, limit: int) -> List[str]:
    """
    現在のGoogle SERPから、外部サイトの結果リンクを複数のセレクタで抽出してフェイルオーバ。
    """
    urls: List[str] = []
    errors: List[str] = []

    # セレクタ候補（経験的に強い順）
    selector_sets = [
        # 1) a:has(h3) … 一般的オーガニック
        ("a:has(h3)", True),
        ("a[jsname][href]", False),    # 取りこぼし救済
        # 2) 直下のカードに付与されることが多い yuRUbf
        ("div.yuRUbf > a", False),
        # 3) g-card内の一般枠など（保険）
        ("div.g a[href]", False),
        # 4) h3の親aを辿る（JS実行）
        #    → 下で JS 実行で追加抽出
    ]

    # まずは CSS ロケータで取れるだけ取る
    for sel, need_wait in selector_sets:
        try:
            if need_wait:
                page.wait_for_selector(sel, timeout=1500)
            loc = page.locator(sel)
            n = min(loc.count(), limit * 3)  # 余裕を取ってからフィルタ
            for i in range(n):
                try:
                    raw = loc.nth(i).get_attribute("href")
                except Exception:
                    raw = None
                href = _normalize_and_filter_href(raw)
                if not href:
                    continue
                urls.append(href)
                # 早期打ち切り
                if len(urls) >= limit:
                    break
            if len(urls) >= limit:
                break
        except (PWTimeout, PWError) as e:
            errors.append(f"{sel}: {type(e).__name__}")
            continue

    # さらに空なら JS で拾う（h3 → 最近傍の a）
    if len(urls) < limit:
        try:
            js = """
            () => {
              const out = [];
              const seen = new Set();
              const hs = document.querySelectorAll('h3');
              for (const h of hs) {
              // 画像やニュースなど特殊枠をなるべく避ける（厳密でなくてOK）
                if (h.closest('[role="region"][aria-label^="ニュース"]')) continue;
                let a = h.closest('a');
                if (!a) {
                  // 直近祖先に a が無ければ周辺を探索
                  const p = h.parentElement;
                  if (p) {
                    const aa = p.querySelectorAll('a[href]');
                    if (aa && aa.length) a = aa[0];
                  }
                }
                if (!a) continue;
                const href = a.getAttribute('href') || '';
                if (!href) continue;
                if (seen.has(href)) continue;
                seen.add(href);
                out.push(href);
              }
              return out;
            }
            """
            cand = page.evaluate(js) or []
            for raw in cand:
                href = _normalize_and_filter_href(raw)
                if not href:
                    continue
                urls.append(href)
                if len(urls) >= limit:
                    break
        except Exception:
            pass

    # 正規化＆重複削除（上限を掛ける）
    return _unique_keep_order(urls, limit)

# ---------- Google 検索（上位URLだけ取る・軽量） ----------

def _search_top_urls(keyword: str, *, limit: int = 6, lang: str = "ja", gl: str = "jp",
                     timeout_ms: int = 18000) -> List[str]:
    """Googleでキーワード検索し、検索結果から外部URLを抽出（フェイルオーバ付き）。"""
    if limit <= 0:
        return []
    q = quote(keyword)
    # 通常Web結果を固定（udm=14）、日本語優先（lr=lang_ja）、パーソナライズ抑制（pws=0）
    base = f"https://www.google.com/search?q={q}&hl={lang}&gl={gl}&num={min(limit, 10)}&pws=0&udm=14&lr=lang_{lang}"
 

    urls: List[str] = []
    with _PWSession(lang=lang, gl=gl) as sess:
        page = sess.new_page()
        # NCRで国別リダイレクトを抑止
        try:
            page.goto("https://www.google.com/ncr", timeout=4000)
        except Exception:
            pass

        def _attempt(visit_url: str) -> List[str]:
            try:
                page.goto(visit_url, timeout=timeout_ms, wait_until="domcontentloaded")
                # 同意/ブロック画面なら空配列にして上位でリトライ判断
                title_text = page.title() or ""
                body_text = page.locator("body").inner_text(timeout=1000) if page.locator("body").count() else ""
                if _is_consent_or_block_page(title_text, body_text):
                    return []
                # 検索結果本体のコンテナが来るのを待つ（柔らかく）
                try:
                    page.wait_for_selector("div#search", timeout=1000)
                except Exception:
                    pass
                links = _extract_result_links(page, limit=limit)
                # 0件なら、何が表示されていたかをスナップショット保存
                if not links:
                    try:
                        t = _NOW()
                        html_path = os.path.join(_DBG_DIR, f"serp_{t}.html")
                        txt_path  = os.path.join(_DBG_DIR, f"serp_{t}.txt")
                        with open(html_path, "w", encoding="utf-8") as f:
                            f.write(page.content() or "")
                        counts = {
                            "a_has_h3": page.locator("a:has(h3)").count(),
                            "jsname_a": page.locator("a[jsname][href]").count(),
                            "yuRUbf_a": page.locator("div.yuRUbf > a").count(),
                            "div_g_a" : page.locator("div.g a[href]").count(),
                            "h3_total": page.locator("h3").count(),
                        }
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"title={title_text}\nurl={visit_url}\ncounts={counts}\n")
                        print(f"[SERP-DEBUG] 0 links → snapshot saved: {html_path}")
                    except Exception:
                        pass
                return links
            except (PWTimeout, PWError):
                return []

        # 1回目
        urls = _attempt(base)
        # 0件なら、軽い再試行（hl固定・num=10・pws=0は維持）
        if not urls:
            urls = _attempt(base + "&source=hp")
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
