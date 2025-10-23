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
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlparse, parse_qs, unquote

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright.sync_api import Error as PWError
import requests
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

# --- CSE(JSON API) 設定 ---
_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")  # 例: AIzaSyXXXX...
_CSE_CX      = os.environ.get("GOOGLE_CSE_CX")       # 例: d115155f883b1466f
_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# 収集の鮮度(TTL)。同一記事の直近キャッシュがこの期間内なら再収集をスキップ
_CACHE_TTL_DAYS = int(os.environ.get("SERP_CACHE_TTL_DAYS", "14"))
# 同一ドメインから拾う最大件数（多様性確保）
_MAX_PER_DOMAIN = int(os.environ.get("SERP_MAX_PER_DOMAIN", "2"))

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

# ---------- Google Custom Search JSON API（公式） ----------
def _path_depth(url: str) -> int:
    """ドメイン直下=0, /a=1, /a/b=2 ... のようなパス深さを返す。"""
    try:
        p = urlparse(url)
        segs = [s for s in (p.path or "").split("/") if s]
        return len(segs)
    except Exception:
        return 0

_QNA_HINTS = ("faq", "q&a", "q%26a", "よくある質問", "質問", "Q＆A", "Q&A")

def _looks_like_qna_text(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _QNA_HINTS)

def _looks_like_qna_article(article_title: str) -> bool:
    """記事タイトルから『Q&A/FAQ系か』を推定（軽量ヒューリスティック）。"""
    return _looks_like_qna_text(article_title)

def _tokenize_keywords(q: str) -> List[str]:
    """スペース分割の緩いトークナイズ（全角/半角/連続スペース対応）。"""
    q = (q or "").strip().replace("　", " ")
    return [t for t in q.split(" ") if t]

def _is_useful_result(url: str, title: str, snippet: str, *,
                      qna_required: bool, keyword_tokens: List[str]) -> bool:
    """
    役に立たない結果（トップ/LP/無関係）を落とす。
    - 常に: ドメイン直下（path depth==0）は除外
    - Q&A記事モード: タイトル/スニペット/URLのいずれかがQ&A/FAQを示唆しないなら除外
    - 最低限: タイトル/スニペットにキーワードのどれかが1つは含まれる
    """
    # 1) トップ/LPは除外（path深さ0）
    if _path_depth(url) == 0:
        return False

    # 2) Q&A記事ならQ&Aらしさが必要
    if qna_required:
        if not (_looks_like_qna_text(title) or _looks_like_qna_text(snippet) or _looks_like_qna_text(url)):
            return False

    # 3) キーワード関連性（タイトルorスニペットに1語以上含まれる）
    t = (title or "") + " " + (snippet or "")
    if keyword_tokens:
        if not any(tok in t for tok in keyword_tokens):
            return False

    return True

def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _search_top_urls_cse(keyword: str, *, limit: int = 6, lang: str = "ja", gl: str = "jp",
                         qna_required: bool = False, article_title: str = "") -> List[Dict[str, str]]:

    """
    Google公式 Custom Search JSON API を使って上位URLを取得する。
    失敗時は空配列。無料枠: 100クエリ/日（課金未設定なら超過時はエラーで停止）。
    """
    if not _CSE_API_KEY or not _CSE_CX:
        # 設定漏れを明示的に知らせる
        raise RuntimeError("GOOGLE_CSE_API_KEY / GOOGLE_CSE_CX が未設定です。環境変数に設定してください。")

    params = {
        "key": _CSE_API_KEY,
        "cx": _CSE_CX,
        # google.com系を弾きつつ通常の検索語を投げる
        "q": f"{keyword} -site:google.com -site:*.google.com",
        "num": min(10, max(1, limit)),  # API仕様上の上限は10
        "hl": lang,
        # 備考: CSEは 'gl' や 'lr' も一部サポート。ただし結果への影響は限定的。
        "lr": f"lang_{lang}",
        "safe": "off",
        # 二重ガード（google.com除外）
        "siteSearch": "google.com",
        "siteSearchFilter": "e",  # e=exclude
    }
    try:
        resp = requests.get(_CSE_ENDPOINT, params=params, timeout=15)
        data = resp.json() if resp.ok else {}
        items = data.get("items", []) or []
        results: List[Dict[str, str]] = []
        per_domain: Dict[str, int] = {}
        kw_tokens = _tokenize_keywords(keyword)
        for it in items:
            link = (it.get("link") or "").strip()
            title = (it.get("title") or "").strip()
            snippet = (it.get("snippet") or "").strip()
            if not (link.startswith("http://") or link.startswith("https://")):
                continue
            # 無関係/トップ/LPの除外
            if not _is_useful_result(link, title, snippet, qna_required=qna_required, keyword_tokens=kw_tokens):
                continue
            # 同一ドメイン多様性の確保
            d = _netloc(link)
            cnt = per_domain.get(d, 0)
            if cnt >= _MAX_PER_DOMAIN:
                continue
            per_domain[d] = cnt + 1

            results.append({"url": link, "title": title, "snippet": snippet})
            if len(results) >= limit:
                break
        # unique keep order by url
        seen = set()
        uniq: List[Dict[str, str]] = []
        for r in results:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            uniq.append(r)
        if not uniq:
            # 何が返ってきたかの証跡保存（デバッグ用）
            t = _NOW()
            with open(os.path.join(_DBG_DIR, f"cse_{t}.json"), "w", encoding="utf-8") as f:
                f.write(resp.text if resp is not None else "{}")
        return uniq
    except Exception as e:
        # 失敗時は空。ログだけ残す
        try:
            t = _NOW()
            with open(os.path.join(_DBG_DIR, f"cse_error_{t}.txt"), "w", encoding="utf-8") as f:
                f.write(f"error={repr(e)} keyword={keyword}\n")
        except Exception:
            pass
        return []

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
                # JavaScript描画を確実に待つ
                page.goto(visit_url, timeout=timeout_ms, wait_until="load")
                # 追加の描画完了を最大2.5秒だけ待つ（新UIはSPA）
                page.wait_for_timeout(2500)

                # もしまだh3が無ければ role="heading" or aria-level="3" を待つ
                if page.locator("h3").count() == 0:
                    try:
                        page.wait_for_selector("div[role='heading'], [aria-level='3']", timeout=2000)
                    except Exception:
                        pass
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
                # 新UIでは <a jsname> のみ出現することが多いので再取得
                if len(links) == 0:
                    js_links = [a.get_attribute("href") for a in page.locator("a[jsname][href]").element_handles()]
                    from urllib.parse import unquote
                    links = [unquote(x) for x in js_links if x and x.startswith("http")]
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


def _detect_schema_types(page) -> List[str]:
    """簡易：JSON-LDの@typeやmicrodataからFAQ/HowTo/Article等を拾う（過検出を避けて軽めに）。"""
    types = set()
    try:
        # JSON-LDの文字列を抽出して軽く判定
        scripts = page.locator('script[type="application/ld+json"]').all()
        for s in scripts[:6]:  # 安全のため上限
            try:
                txt = s.inner_text() or ""
            except Exception:
                txt = ""
            t = txt.lower()
            if '"faqpage"' in t or "'faqpage'" in t:
                types.add("FAQ")
            if '"howto"' in t or "'howto'" in t:
                types.add("HowTo")
            if '"article"' in t or "'article'" in t:
                types.add("Article")
    except Exception:
        pass
    # microdataの痕跡（緩く）
    try:
        if page.locator('[itemtype*="FAQPage"]').count() > 0:
            types.add("FAQ")
        if page.locator('[itemtype*="HowTo"]').count() > 0:
            types.add("HowTo")
        if page.locator('[itemtype*="Article"]').count() > 0:
            types.add("Article")
    except Exception:
        pass
    return sorted(types)

def _extract_intro_text(page, limit_chars: int = 2000) -> str:
    """本文冒頭テキストを軽量に抽出（main/article/#content/.entry-content を優先）。"""
    candidates = [
        "main", "article", "#content", ".entry-content", ".post-content", ".content",
    ]
    text = ""
    try:
        for sel in candidates:
            loc = page.locator(sel)
            if loc.count() > 0:
                try:
                    t = loc.first.inner_text(timeout=1000) or ""
                except Exception:
                    t = ""
                t = _WS_RE.sub(" ", t).strip()
                if len(t) >= 80:  # あまりに短いのはスキップ
                    text = t
                    break
        if not text:
            # やむなく body 全体から
            try:
                t = page.locator("body").inner_text(timeout=1000) or ""
                text = _WS_RE.sub(" ", t).strip()
            except Exception:
                text = ""
    except Exception:
        text = ""
    return text[:limit_chars]

def _fetch_page_outline(url: str, *, timeout_ms: int = 18000, lang: str = "ja", gl: str = "jp") -> Dict[str, Any]:
    """
    URLを開いて見出し/冒頭/構造化データ/シグナルを抽出。
    失敗しても {"url": url, "h": []} を返す。
    """
    with _PWSession(lang=lang, gl=gl) as sess:
        page = sess.new_page()
        try:
            resp = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            status = None
            try:
                status = resp.status if resp else None
            except Exception:
                status = None

            heads = _extract_headings_from_dom(page, timeout_ms=max(8000, timeout_ms // 2))
            if not heads:
                html = page.content()
                heads = _extract_headings_from_html(html or "")

            intro = _extract_intro_text(page, limit_chars=2000)
            schema_types = _detect_schema_types(page)

            # 軽量シグナル
            wc = len((intro or "").split())
            has_table = False
            try:
                has_table = page.locator("table").count() > 0
            except Exception:
                has_table = False
            signals = {
                "word_count": wc,
                "has_faq": "FAQ" in schema_types,
                "has_howto": "HowTo" in schema_types,
                "has_table": bool(has_table),
            }
            return {"url": url, "h": heads, "intro": intro, "schema": schema_types, "signals": signals, "http": {"status": status}}
        except Exception:
            return {"url": url, "h": []}

# ---------- 収集・保存パイプライン ----------

def collect_serp_outlines_for_keyword(keyword: str, *, limit: int = 6,
                                      lang: str = "ja", gl: str = "jp",
                                      qna_required: bool = False, article_title: str = "") -> List[Dict[str, Any]]:

    """
    キーワードでGoogle検索（CSE JSON API） → 上位URLを巡回 → 各ページのH2/H3を抽出して返却。
    （SERPのHTMLスクレイピングは行わない）
    戻り値: [{url, h:[...]}, ...]
    """
    results = _search_top_urls_cse(keyword, limit=limit, lang=lang, gl=gl,
                                   qna_required=qna_required, article_title=article_title)
    outlines: List[Dict[str, Any]] = []
    # 1URLずつ巡回（並列はリソース重いので避ける）
    for r in results:
        detail = _fetch_page_outline(r["url"], lang=lang, gl=gl)
        # CSEの title/snippet を付与（後方互換のため任意項目）
        detail["title"] = r.get("title") or ""
        detail["snippet"] = r.get("snippet") or ""
        outlines.append(detail)
    return outlines


def cache_outlines(article_id: int, outlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SerpOutlineCacheに保存（新規行として追加）。
    既存キャッシュを消さず、最新を参照したい場合は executed_at/fetched_at の新しいレコードを使う想定。
    """
    rec = SerpOutlineCache(article_id=article_id, outlines=outlines)
    db.session.add(rec)
    db.session.commit()
    return {"article_id": article_id, "saved_count": len(outlines), "cache_id": rec.id}


def _is_recent_cache(rec: SerpOutlineCache) -> bool:
    """fetched_at が直近 TTL 日以内なら True（UTC/ローカル混在に強く）。"""
    try:
        if not rec or not rec.fetched_at:
            return False
        ft = rec.fetched_at
        # ft がタイムゾーン無しなら UTC とみなす／ありなら UTC に揃える
        if ft.tzinfo is None:
            ft_utc = ft.replace(tzinfo=timezone.utc)
        else:
            ft_utc = ft.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        return ft_utc >= now_utc - timedelta(days=_CACHE_TTL_DAYS)
    except Exception:
        return False

def collect_and_cache_for_article(article_id: int, *, limit: int = 6,
                                  lang: str = "ja", gl: str = "jp",
                                  force: bool = False) -> Dict[str, Any]:
    """
    Article.id から keyword（なければ title）を使ってSERP収集→キャッシュへ保存。
    """
    art = db.session.get(Article, article_id)
    if not art:
        return {"ok": False, "error": f"article_id={article_id} not found"}

    query = (art.keyword or art.title or "").strip()
    if not query:
        return {"ok": False, "error": "no keyword or title to search"}
    
    # 直近キャッシュが新鮮ならスキップ（API枠節約）
    try:
        latest: Optional[SerpOutlineCache] = (
            db.session.query(SerpOutlineCache)
            .filter(SerpOutlineCache.article_id == article_id)
            .order_by(SerpOutlineCache.fetched_at.desc())
            .first()
        )
    except Exception:
        latest = None

    if not force and latest and _is_recent_cache(latest):
        return {"ok": True, "query": query, "skipped": "recent_cache", "cache_id": latest.id, "saved_count": len(latest.outlines or [])}


    # 記事タイトルからQ&A/FAQっぽさを推定し、必要ならQ&A系ページ以外を除外
    qna_required = _looks_like_qna_article(art.title or "")
    outlines = collect_serp_outlines_for_keyword(query, limit=limit, lang=lang, gl=gl,
                                                 qna_required=qna_required, article_title=art.title or "")
    saved = cache_outlines(article_id, outlines)
    return {"ok": True, "query": query, **saved}
