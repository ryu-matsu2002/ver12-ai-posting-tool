import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re as _re
from urllib.parse import urlparse

from app import db
from app.models import ExternalBlogAccount
from app.enums import BlogType

logger = logging.getLogger(__name__)

# 可能ならサインアップ時の CAPTCHA 手動入力ツールを流用（存在しなければフォールバックへ）
try:
    from app.services.blog_signup.livedoor_signup import (
        prepare_captcha as ld_prepare_captcha,
        submit_captcha as ld_submit_captcha,
    )
except Exception:
    ld_prepare_captcha = None
    ld_submit_captcha = None

# 可能なら pwctl（セッションIDや一時保存ディレクトリの流儀を合わせるため）
try:
    from app.services.pw_controller import pwctl  # noqa
except Exception:
    pwctl = None

# ビルド識別（デプロイ反映チェック用）
BUILD_TAG = "2025-08-28 create-page-captcha+human-tool+fs-fallback+welcome_url_extract+public_url+prefer_userid_sub"
logger.info(f"[LD-Recover] loaded build {BUILD_TAG}")


async def _save_shot(page, prefix: str) -> tuple[str, str]:
    """
    現在ページを /tmp/{prefix}_{ts}.{png,html} で保存してパスを返す。
    失敗時は full_page=False にフォールバック。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = f"/tmp/{prefix}_{ts}.png"
    html = f"/tmp/{prefix}_{ts}.html"
    try:
        await page.screenshot(path=png, full_page=True)
    except Exception:
        try:
            await page.screenshot(path=png)
        except Exception:
            pass
    try:
        Path(html).write_text(await page.content(), encoding="utf-8")
    except Exception:
        pass
    logger.info("[LD-Recover] dump saved: %s , %s", png, html)
    return png, html


# ─────────────────────────────────────────────
# 安定インデックス・文字種判定・正規化などのユーティリティ
# ─────────────────────────────────────────────
def _deterministic_index(salt: str, n: int) -> int:
    """
    salt（文字列）から 0..n-1 の安定インデックスを決める。
    - ランタイム/プロセスを跨いでも同じ salt, n なら同じ値
    - n <= 0 の場合は 0
    """
    if n <= 0:
        return 0
    # 32bit rolling hash（Python の hash は起動ごとに変わるため使わない）
    acc = 0
    for ch in str(salt):
        acc = (acc * 131 + ord(ch)) & 0xFFFFFFFF
    return acc % n


def _has_cjk(s: str) -> bool:
    return bool(_re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", s or ""))


def _norm(s: str) -> str:
    """比較用：空白/記号を落として小文字化"""
    s = (s or "").lower()
    s = _re.sub(r"[\s\-_／|｜/・]+", "", s)
    return s


def _domain_tokens(url: str) -> list[str]:
    """ドメインを単語に分割（tld等は除外）"""
    try:
        netloc = urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    parts = [p for p in netloc.split(".") if p and p not in ("www", "com", "jp", "net", "org", "co")]
    words = []
    for p in parts:
        words.extend([w for w in p.replace("_", "-").split("-") if w])
    return words


# ─────────────────────────────────────────────
# サイト名トークン化・ジャンル推定・日本語タイトル生成
# ─────────────────────────────────────────────
STOPWORDS_JP = {
    "株式会社", "有限会社", "合同会社", "公式", "オフィシャル", "ブログ", "サイト", "ホームページ",
    "ショップ", "ストア", "サービス", "工房", "教室", "情報", "案内", "チャンネル", "通信", "マガジン"
}
STOPWORDS_EN = {
    "inc", "ltd", "llc", "official", "blog", "site", "homepage", "shop", "store",
    "service", "studio", "channel", "magazine", "info", "news"
}


def _name_tokens(name: str) -> list[str]:
    """サイト名を雑にトークン化（日本語/英語混在対応・記号で分割）"""
    if not name:
        return []
    parts = _re.split(r"[\s\u3000\-/＿_・|｜／]+", str(name))
    toks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 記号を除去しすぎない程度に掃除
        p = _re.sub(r"[^\w\u3040-\u30FF\u3400-\u9FFFー]+", "", p)
        if p:
            toks.append(p)
    return toks


def _keyword_seed_from_site(site) -> tuple[str | None, bool]:
    """
    サイト名/URLから「1語」を安定に選ぶ。
    戻り値: (seed, is_jp)  /  抽出できなければ (None, False)
    """
    name = (getattr(site, "name", "") or "").strip()
    url = (getattr(site, "url", "") or "").strip()
    salt = f"{getattr(site, 'id', '')}-{name}-{url}"

    name_toks = _name_tokens(name)
    domain_toks = _domain_tokens(url)

    # 日本語と英語で候補を分ける
    jp_cands = [t for t in name_toks if _has_cjk(t) and t not in STOPWORDS_JP]
    en_cands = [t for t in (name_toks + domain_toks) if not _has_cjk(t)]
    en_cands = [t for t in en_cands if t.lower() not in STOPWORDS_EN]

    # 長さフィルタ（1文字や長すぎは除外）
    jp_cands = [t for t in jp_cands if 2 <= len(t) <= 12]
    en_cands = [t for t in en_cands if 2 <= len(t) <= 15]

    # 同一サイトでは安定して同じ語を選ぶ（塩＝site.id+name+url）
    def _pick(stable_list: list[str]) -> str | None:
        if not stable_list:
            return None
        idx = _deterministic_index(salt, len(stable_list))
        return stable_list[idx]

    seed = _pick(jp_cands) or _pick(en_cands)
    if seed:
        return seed, _has_cjk(seed)
    return None, False


def _guess_genre(site) -> tuple[str, bool]:
    """
    サイトからジャンル語(日本語/英語)と日本語フラグを推定。
    1) 明示属性（primary_genre_name / genre_name / genre.name など）
    2) site.name / site.url の語からヒューリスティック
    """
    # 1) 明示属性
    for attr in ("primary_genre_name", "genre_name", "genre", "main_genre", "category", "category_name"):
        v = getattr(site, attr, None)
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            return txt, _has_cjk(txt)
        name = getattr(v, "name", None)
        if isinstance(name, str) and name.strip():
            txt = name.strip()
            return txt, _has_cjk(txt)

    # 2) ヒューリスティック
    name = (getattr(site, "name", "") or "")
    url = (getattr(site, "url", "") or "")
    txt = (name + " " + url).lower()
    toks = set(_domain_tokens(url))

    JP = [
        ("ピラティス", ("pilates", "ピラティス", "yoga", "体幹", "姿勢", "fitness", "stretch")),
        ("留学", ("studyabroad", "abroad", "留学", "ielts", "toefl", "海外", "study")),
        ("旅行", ("travel", "trip", "観光", "hotel", "onsen", "温泉", "tour")),
        ("美容", ("beauty", "esthetic", "skin", "hair", "美容", "コスメ", "メイク")),
        ("ビジネス", ("business", "marketing", "sales", "seo", "経営", "起業", "副業")),
    ]
    for label, keys in JP:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, True

    EN = [
        ("Pilates", ("pilates", "yoga", "fitness", "posture", "stretch")),
        ("Study Abroad", ("studyabroad", "abroad", "study", "ielts", "toefl")),
        ("Travel", ("travel", "trip", "hotel", "onsen", "tour")),
        ("Beauty", ("beauty", "esthetic", "skin", "hair", "cosme", "makeup")),
        ("Business", ("business", "marketing", "sales", "seo", "startup")),
    ]
    for label, keys in EN:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, False

    # どれにも該当しなければ汎用
    return ("日々", _has_cjk(name) or _has_cjk(url))


def _too_similar_to_site(title: str, site) -> bool:
    """
    タイトルがサイト名/ドメイン由来語と似すぎなら True。
    - 正規化同士の完全一致
    - 片方がもう片方を包含
    - ドメイン語幹（tokens）が含まれる/含まれる
    """
    t = _norm(title)
    site_name = (getattr(site, "name", "") or "")
    site_url = (getattr(site, "url", "") or "")
    n = _norm(site_name)

    if not t:
        return True

    # 完全一致 / 包含
    if t == n or (t and n and (t in n or n in t)):
        return True

    # ドメイン語幹との照合
    toks = set(_domain_tokens(site_url))
    toks |= {w for w in _name_tokens(site_name) if not _has_cjk(w)}  # 英字トークンも禁止寄り
    toks = {_norm(w) for w in toks if w}

    for w in toks:
        if not w:
            continue
        if w in t or t in w:
            return True

    return False


def _templates_jp(topic: str) -> list[str]:
    base = (topic or "").strip() or "日々"
    return [
        f"{base}ブログ",
        f"{base}ブログ日記",
        f"{base}のブログ",
        f"{base}の記録ブログ",
        f"{base}の暮らしブログ",
        f"{base}のメモ帳",
        f"{base}の覚え書き",
        f"{base}のジャーナル",
        f"{base}手帖",
        f"{base}ノート",
        f"{base}の小部屋",
        f"{base}ログ",
    ]


def _templates_en(topic: str) -> list[str]:
    base = topic.strip() or "Notes"
    return [f"{base} Blog"]  # ダミー（呼ばれない想定）


def _japanese_base_word(site) -> str:
    """
    1) まずジャンル推定で日本語ラベルを取得（ピラティス/旅行/美容/ビジネス…）
    2) 取れなければ「日々」
    ※ “サイト名そのもの”は使わない（似すぎ回避）
    """
    topic, is_jp = _guess_genre(site)
    if _has_cjk(topic):
        return topic.strip()
    return "日々"


def _craft_blog_title(site) -> str:
    """
    仕様：
      - 生成結果は必ず“日本語”
      - 元サイト名やドメインに“似すぎない”
      - かならず“ブログっぽい”語尾（～ブログ 等）を含める
      - 同一サイトでは決定論的に安定
    """
    site_name = (getattr(site, "name", "") or "").strip()
    site_url = (getattr(site, "url", "") or "").strip()
    salt = f"{getattr(site, 'id', '')}-{site_name}-{site_url}"

    # 1) 日本語のベース語（ジャンルラベル優先）
    base_word = _japanese_base_word(site)

    # 2) 候補群（日本語テンプレのみ）
    cands = _templates_jp(base_word)

    # 3) “似すぎ”禁止セット（強め）
    banned_equal = {_norm(site_name)}
    banned_equal.update(_norm(w) for w in _domain_tokens(site_url))

    def acceptable(title: str) -> bool:
        # 空/None
        if not title or not title.strip():
            return False
        # 完全一致の禁止
        if _norm(title) in banned_equal:
            return False
        # 類似（包含/ドメイン語幹）の禁止
        if _too_similar_to_site(title, site):
            return False
        # 日本語強制
        if not _has_cjk(title):
            return False
        # “ブログっぽさ”担保
        if "ブログ" not in title:
            return False
        return True

    # 4) salt で開始位置を決め、順送りで最初に通ったもの
    idx = _deterministic_index(salt, len(cands))
    for i in range(len(cands)):
        title = cands[(idx + i) % len(cands)]
        if acceptable(title):
            return title[:48]

    # 5) 最終フォールバック
    fallback = [f"{base_word}ブログ", f"{base_word}のブログ", "日々ブログ", "小さなブログ記録"]
    return fallback[_deterministic_index(salt, len(fallback))][:48]


# ─────────────────────────────────────────────
# blog_id スラッグ生成・入力欄設定ヘルパ
# ─────────────────────────────────────────────
def _slugify_ascii(s: str) -> str:
    """日本語/記号混じり → 半角英数とハイフンの短いスラッグ（先頭英字、長さ20程度）"""
    try:
        from unidecode import unidecode
    except Exception:
        def unidecode(x): return x
    if not s:
        s = "blog"
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ")
    s = _re.sub(r"[^a-z0-9]+", "-", s)
    s = _re.sub(r"-{2,}", "-", s).strip("-")
    if s and s[0].isdigit():
        s = "blog-" + s
    if not s:
        s = "blog"
    s = s[:20]
    if len(s) < 3:
        s = (s + "-blog")[:20]
    return s


async def _try_set_desired_blog_id(page, desired: str) -> bool:
    """
    ブログ作成画面で希望 blog_id / サブドメインを入力する。
    画面ごとの違いに備えて複数セレクタを順に試す。
    成功/入力欄が見つからない場合は True、明確な失敗で False を返す。
    """
    # まず従来セレクタ（ID型）
    id_selectors = [
        '#blogId',
        'input[name="blog_id"]',
        'input[name="livedoor_blog_id"]',
        'input[name="blogId"]',
        'input#livedoor_blog_id',
        'input[placeholder*="ブログURL"]',
    ]
    try:
        for sel in id_selectors:
            try:
                if await page.locator(sel).count() > 0:
                    await page.fill(sel, desired)
                    return True
            except Exception:
                continue
    except Exception:
        pass

    # サブドメイン型（今回ヒットしているやつ）
    try:
        sub_loc = None
        if await page.locator('#sub').count() > 0:
            sub_loc = page.locator('#sub').first
        elif await page.locator('input[name="sub"]').count() > 0:
            sub_loc = page.locator('input[name="sub"]').first

        if sub_loc:
            # base があれば最初の有効optionを選ぶ（既定でOKならそのまま）
            try:
                if await page.locator('#base').count() > 0:
                    base = page.locator('#base').first
                    # 選択済みでも change を発火させ domaincheck を走らせる
                    await base.evaluate("(el)=>el.dispatchEvent(new Event('change', {bubbles:true}))")
            except Exception:
                pass

            # 入力とイベント発火（htmxのdomaincheck用）
            try:
                await sub_loc.fill("")
            except Exception:
                try:
                    await sub_loc.click()
                    await sub_loc.press("Control+A")
                    await sub_loc.press("Delete")
                except Exception:
                    pass
            await sub_loc.fill(desired)
            try:
                await sub_loc.evaluate("(el)=>el.dispatchEvent(new Event('keyup', {bubbles:true}))")
            except Exception:
                pass

            # domaincheck の結果を短時間待つ（OK/NGはこの後のリトライ側で判定）
            try:
                await page.wait_for_timeout(1200)  # htmx domaincheck 余裕を持たせる
            except Exception:
                pass
            return True
    except Exception:
        pass

    # どれも見つからない → 致命ではない
    return True



# ─────────────────────────────────────────────
# 追加：フレーム横断・同意チェック・エラーテキスト採取
# ─────────────────────────────────────────────
async def _maybe_close_overlays(page):
    selectors = [
        'button#iubenda-cs-accept-btn',
        'button#iubenda-cs-accept',
        'button:has-text("同意")',
        'button:has-text("許可")',
        'button:has-text("OK")',
        '.cookie-accept', '.cookie-consent-accept',
        '.modal-footer button:has-text("閉じる")',
        'div[role="dialog"] button:has-text("OK")',
    ]
    for sel in selectors:
        try:
            if await page.locator(sel).first.is_visible():
                await page.locator(sel).first.click(timeout=1000)
        except Exception:
            pass
    # 透明オーバーレイの一般除去
    try:
        await page.evaluate("""
            (() => {
              const blocks = Array.from(document.querySelectorAll('div,section'))
                .filter(n => {
                  const s = getComputedStyle(n);
                  if (!s) return false;
                  const r = n.getBoundingClientRect();
                  return r.width>300 && r.height>200 &&
                         s.position !== 'static' &&
                         parseFloat(s.zIndex||'0') >= 1000 &&
                         s.pointerEvents !== 'none' &&
                         (s.backgroundColor && s.backgroundColor !== 'rgba(0, 0, 0, 0)');
                });
              blocks.slice(0,3).forEach(n => n.style.pointerEvents='none');
            })();
        """)
    except Exception:
        pass


async def _maybe_accept_terms(page) -> bool:
    """利用規約の同意チェックがあればON。チェック状態が変わったら True を返す。"""
    sels = [
        'input[type="checkbox"][name*="agree"]',
        'input#agree', 'input#agreement', 'input#termsAgree',
        'input#accept-terms', 'input[name="agreement"]', 'input[name="accept"]'
    ]
    changed = False
    for sel in sels:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                try:
                    await loc.check()
                    changed = True
                except Exception:
                    # check() で失敗した際の直接操作フォールバック
                    try:
                        handle = await loc.element_handle()
                        if handle:
                            await page.evaluate(
                                "(el)=>{if(!el.checked){el.checked=true;} el.dispatchEvent(new Event('change',{bubbles:true}))}", handle
                            )
                            changed = True
                    except Exception:
                        pass
                if changed:
                    logger.info("[LD-Recover] ✅ 規約同意チェック: %s", sel)
                    break
        except Exception:
            pass
    return changed


async def _has_blog_id_input(page) -> bool:
    for sel in [
        '#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]',
        'input[name="blogId"]', 'input#livedoor_blog_id', 'input[placeholder*="ブログURL"]',  # ← ここにカンマ
        '#sub', 'input[name="sub"]'   # ← これが生きる
    ]:
        try:
            if await page.locator(sel).count() > 0:
                return True
        except Exception:
            pass
    return False



async def _log_inline_errors(page):
    """画面上の代表的なエラーメッセージを収集してログ出力"""
    sels = [
        '.error', '.error-message', '.errors li', 'p.error', 'span.error',
        '.alert-danger', '.alert.alert-danger', 'div.errorMessage', 'li.error',
        'div.formError', 'div#notice .error', 'div.notice.error'
    ]
    texts = []
    for sel in sels:
        try:
            loc = page.locator(sel)
            cnt = await loc.count()
            for i in range(min(cnt, 10)):
                t = (await loc.nth(i).inner_text()).strip()
                if t:
                    texts.append(t.replace("\n", " "))
        except Exception:
            pass
    if texts:
        logger.warning("[LD-Recover] inline errors: %s", " | ".join(texts[:5]))


async def _find_in_any_frame(page, selectors, timeout_ms=15000):
    """全フレーム走査。最初に見つかったframeとselectorを返す。"""
    logger.info("[LD-Recover] frame-scan start selectors=%s timeout=%sms", selectors[:2], timeout_ms)
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
    while asyncio.get_event_loop().time() < deadline:
        try:
            for fr in page.frames:
                for sel in selectors:
                    try:
                        if await fr.locator(sel).count() > 0:
                            logger.info("[LD-Recover] frame-scan hit: frame=%s sel=%s", getattr(fr, 'url', None), sel)
                            return fr, sel
                    except Exception:
                        continue
        except Exception:
            pass
        await asyncio.sleep(0.25)
    logger.warning("[LD-Recover] frame-scan timeout selectors=%s", selectors[:3])
    return None, None


async def _wait_enabled_and_click(page, locator, *, timeout=8000, label_for_log=""):
    try:
        await locator.wait_for(state="visible", timeout=timeout)
    except Exception:
        try:
            await locator.wait_for(state="attached", timeout=int(timeout/2))
        except Exception:
            pass

    # ★ ElementHandle を取得
    try:
        handle = await locator.element_handle()
    except Exception:
        handle = None

    # enabled/表示状態
    if handle:
        try:
            await page.wait_for_function(
                "(el) => el && !el.disabled && el.offsetParent !== null",
                arg=handle, timeout=timeout
            )
        except Exception:
            pass

    try:
        await locator.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass
    try:
        await locator.focus()
    except Exception:
        pass

    # クリック多段
    try:
        await locator.click(timeout=timeout)
        logger.info("[LD-Recover] clicked %s (normal)", label_for_log or "")
        return True
    except Exception:
        try:
            await locator.click(timeout=timeout, force=True)
            logger.info("[LD-Recover] clicked %s (force)", label_for_log or "")
            return True
        except Exception:
            if handle:
                try:
                    await page.evaluate("(el)=>el.click()", handle)
                    logger.info("[LD-Recover] clicked %s (evaluate)", label_for_log or "")
                    return True
                except Exception:
                    pass
            logger.warning("[LD-Recover] click failed %s", label_for_log, exc_info=True)
            return False


# ─────────────────────────────────────────────
# ブログ作成ページ：タイトル入力＆送信
# ─────────────────────────────────────────────
async def _set_title_and_submit(page, desired_title: str) -> bool:
    """
    タイトル欄と『ブログを作成する』を見つけて送信。
    1) まずメインフレームで厳密に待つ
    2) ダメなら全フレーム走査
    3) クリック前にスクロール＆フォーカス
    4) クリックは expect_navigation を優先し、失敗時は多段フォールバック
    5) どこで失敗してもログ＋スクショ
    """
    await _maybe_close_overlays(page)

    title_primary = ['#blogTitle', 'input[name="title"]']
    title_fallback = [
        '#blogTitle', 'input#blogTitle', 'input[name="title"]',
        'input#title', 'input[name="blogTitle"]', 'input[name="blog_title"]',
        'input[placeholder*="ブログ"]', 'input[placeholder*="タイトル"]',
    ]
    create_btn_sels = [
        'input[type="submit"][value="ブログを作成する"]',
        'input[type="submit"][value*="ブログを作成"]',
        'input[type="submit"][value*="ブログ作成"]',
        'input[type="submit"][value*="作成"]',
        'input[type="submit"][value*="登録"]',
        '#commit-button',
        'button[type="submit"]',
        'button:has-text("ブログを作成")',
        'button:has-text("ブログを開設")',
        'button:has-text("ブログを始める")',
        'button:has-text("作成")',
        'button:has-text("登録")',
        'a.button:has-text("ブログを作成")',
        'a:has-text("ブログを作成")',
        'a:has-text("ブログを開設")',
        'a:has-text("ブログを始める")',
    ]

    # --- 1) メインフレームで厳密に待つ ---
    try:
        logger.info("[LD-Recover] タイトル設定＆送信開始（main-frame first）")
        found = False
        for sel in title_primary:
            try:
                await page.wait_for_selector(sel, state="visible", timeout=20000)
                el = page.locator(sel).first
                # クリア
                try:
                    await el.fill("")
                except Exception:
                    try:
                        await el.click()
                        await el.press("Control+A")
                        await el.press("Delete")
                    except Exception:
                        pass
                await el.fill(desired_title)
                logger.info("[LD-Recover] ブログタイトルを設定: %s (%s)", desired_title, sel)
                found = True
                break
            except Exception:
                continue

        if not found:
            # --- 2) 全フレーム走査 ---
            fr, sel = await _find_in_any_frame(page, title_fallback, timeout_ms=20000)
            if not fr:
                logger.warning("[LD-Recover] タイトル入力欄が見つからない（DOM/iframe変更の可能性）")
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    await page.screenshot(path=f"/tmp/ld_title_not_found_{ts}.png", full_page=True)
                    logger.info("[LD-Recover] dump: /tmp/ld_title_not_found_%s.png", ts)
                except Exception:
                    pass
                return False

            el = fr.locator(sel).first
            try:
                await el.fill("")
            except Exception:
                try:
                    await el.click()
                    await el.press("Control+A")
                    await el.press("Delete")
                except Exception:
                    pass
            await el.fill(desired_title)
            logger.info("[LD-Recover] ブログタイトルを設定(frame): %s (%s)", desired_title, sel)

    except Exception:
        logger.warning("[LD-Recover] タイトル入力に失敗", exc_info=True)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            await page.screenshot(path=f"/tmp/ld_title_fill_error_{ts}.png", full_page=True)
            logger.info("[LD-Recover] dump: /tmp/ld_title_fill_error_%s.png", ts)
        except Exception:
            pass
        return False

    # --- 3) 作成ボタンを探してクリック ---
    try:
        # まずメインフレーム
        btn = None
        btn_sel = None
        for sel in create_btn_sels:
            loc = page.locator(sel).first
            try:
                if await loc.count() > 0:
                    btn = loc
                    btn_sel = sel
                    break
            except Exception:
                continue
        if btn is None:
            # 全フレーム
            fr_btn, btn_sel = await _find_in_any_frame(page, create_btn_sels, timeout_ms=10000)
            if not fr_btn:
                logger.warning("[LD-Recover] 作成ボタンが見つからない（UI変更の可能性）")
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    await page.screenshot(path=f"/tmp/ld_button_not_found_{ts}.png", full_page=True)
                    logger.info("[LD-Recover] dump: /tmp/ld_button_not_found_%s.png", ts)
                except Exception:
                    pass
                return False
            btn = fr_btn.locator(btn_sel).first

        # ★クリック直前の証跡
        try:
            await _save_shot(page, "ld_create_before_submit")
        except Exception:
            pass

        # まずは expect_navigation で遷移イベントを掴む
        try:
            async with page.expect_navigation(wait_until="load", timeout=30000):
                try:
                    await btn.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                try:
                    await btn.focus()
                except Exception:
                    pass
                await btn.click()
            logger.info("[LD-Recover] 『ブログを作成』ボタンをクリック: %s (expect_navigation)", btn_sel)
        except Exception:
            # フォールバッククリック
            logger.info("[LD-Recover] expect_navigation を掴めず。フォールバッククリックに切替")
            clicked = await _wait_enabled_and_click(page, btn, timeout=8000, label_for_log=f"create-button {btn_sel}")
            if not clicked:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    await page.screenshot(path=f"/tmp/ld_button_click_error_{ts}.png", full_page=True)
                    logger.info("[LD-Recover] dump: /tmp/ld_button_click_error_%s.png", ts)
                except Exception:
                    pass
                return False

        # 追加でロード待ち（遷移しないUIでも描画更新を待つ）
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

    except Exception:
        logger.warning("[LD-Recover] 作成ボタンクリック処理で例外", exc_info=True)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            await page.screenshot(path=f"/tmp/ld_button_click_exception_{ts}.png", full_page=True)
            logger.info("[LD-Recover] dump: /tmp/ld_button_click_exception_%s.png", ts)
        except Exception:
            pass
        return False

    return True


# ─────────────────────────────────────────────
# 作成ページ CAPTCHA 検出・処理（人力ツール優先／FSフォールバック）
# ─────────────────────────────────────────────
async def _detect_create_captcha(page) -> tuple[bool, str | None]:
    try:
        img_sel = '#captcha_image, #captcha-img, img.captcha'  # 実体＋互換
        box_sel = 'input[name="captcha_code"], input[name="captcha"], #captcha'
        has_img = await page.locator(img_sel).first.count() > 0
        has_box = await page.locator(box_sel).first.count() > 0
        if not (has_img and has_box):
            return False, None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/tmp/ld_create_captcha_{ts}.png"
        try:
            await page.locator(img_sel).first.screenshot(path=path)
        except Exception:
            await page.screenshot(path=path, full_page=True)
        logger.info("[LD-Recover] create CAPTCHA captured: %s", path)
        return True, path
    except Exception:
        return False, None



async def _fill_captcha_and_submit(page, text: str) -> bool:
    try:
        box_sel = 'input[name="captcha_code"], input[name="captcha"], #captcha'
        loc = page.locator(box_sel).first
        await loc.fill("")
        await loc.fill(text)

        # 送信ボタン（代表選抜）
        btn = page.locator('input[type="submit"][value="ブログを作成する"]').first
        if await btn.count() == 0:
            btn = page.locator(
                'button[type="submit"], input[type="submit"][value*="作成"], input[type="submit"][value*="登録"]'
            ).first

        await _save_shot(page, "ld_create_before_submit_with_captcha")
        try:
            async with page.expect_navigation(wait_until="load", timeout=15000):
                await btn.click()
        except Exception:
            await _wait_enabled_and_click(page, btn, timeout=8000, label_for_log="create-after-captcha")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
        return True
    except Exception:
        logger.warning("[LD-Recover] CAPTCHA 入力→送信に失敗", exc_info=True)
        return False



async def _wait_success_after_submit(page) -> tuple[bool, str | None]:
    """/welcome 遷移 or 成功導線検出を待って成功可否と URL 由来 blog_id を返す。"""
    try:
        await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=12000)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass

    # 文言 or 導線
    try:
        await page.wait_for_selector('text=ブログの作成が完了しました', timeout=6000)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass
    try:
        await page.wait_for_selector('text=ブログの作成が完了しました！', timeout=3000)
        return True, _extract_blog_id_from_url(page.url)
    except Exception:
        pass

    fr, _ = await _find_in_any_frame(
        page,
        ['a:has-text("最初のブログを書く")', 'a.button:has-text("はじめての投稿")', ':has-text("ブログが作成されました")'],
        timeout_ms=6000
    )
    if fr:
        return True, _extract_blog_id_from_url(page.url)

    return False, None


import inspect

async def _human_tool_captcha_flow(
    page,
    image_path: str,
    *,
    livedoor_id: str | None = None,
    password: str | None = None,
) -> bool:
    """
    サインアップ時の手動入力ツールを優先利用。
    - ld_prepare_captcha / ld_submit_captcha が使える場合はそれを使う
    - 使えない場合は FS 監視フォールバック
    """
    # 1) まずは既存ツール（関数が見つかれば使う）
    if ld_prepare_captcha and ld_submit_captcha:
        try:
            logger.info("[LD-Recover] using signup captcha tool on create-page")
            # 既存実装の引数差異を吸収
            def _callable_with(args_map, fn):
                sig = inspect.signature(fn)
                kwargs = {}
                for name in sig.parameters.keys():
                    if name in args_map and args_map[name] is not None:
                        kwargs[name] = args_map[name]
                return kwargs
            args_map = {
                "page": page,
                "livedoor_id": livedoor_id,
                "password": password,
            }
            await ld_prepare_captcha(**_callable_with(args_map, ld_prepare_captcha))
            await ld_submit_captcha(**_callable_with(args_map, ld_submit_captcha))
            return True
        except Exception:
            logger.warning("[LD-Recover] signup captcha tool failed; fallback to FS watcher", exc_info=True)

    # 2) フォールバック：/tmp を監視して人間が置く回答ファイルを待つ
    try:
        ans_dir = Path("/tmp/captcha_answers")
        ans_dir.mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem  # ld_create_captcha_yyyymmdd_hhmmss
        ans_file = ans_dir / f"{base}.txt"

        # ヒントファイル（UI/オペレータ向け）
        hint_file = ans_dir / f"{base}.readme"
        hint = (
            f"[LD-Recover] 手動CAPTCHA回答の受け付け\n"
            f"- 画像: {image_path}\n"
            f"- 回答ファイルにテキストで解答を保存してください: {ans_file}\n"
        )
        try:
            hint_file.write_text(hint, encoding="utf-8")
        except Exception:
            pass

        logger.info("[LD-Recover] waiting human answer file: %s (up to 180s)", ans_file)
        for _ in range(180):  # 最大180秒
            if ans_file.exists():
                try:
                    text = ans_file.read_text(encoding="utf-8").strip()
                except Exception:
                    text = ""
                if text:
                    logger.info("[LD-Recover] got manual captcha answer: %s (len=%d)", text, len(text))
                    await _fill_captcha_and_submit(page, text)
                    return True
                else:
                    logger.warning("[LD-Recover] answer file empty, keep waiting: %s", ans_file)
            await asyncio.sleep(1.0)
        logger.warning("[LD-Recover] timeout waiting for human captcha answer")
        return False
    except Exception:
        logger.warning("[LD-Recover] FS watcher fallback failed", exc_info=True)
        return False


# ─────────────────────────────────────────────
# メイン：ブログ作成→AtomPub APIキー取得
# ─────────────────────────────────────────────
def _extract_blog_id_from_url(url: str) -> str | None:
    try:
        m = _re.search(r"/blog/([^/]+)/", url)
        return m.group(1) if m else None
    except Exception:
        return None

async def _extract_public_url(page) -> str | None:
    # 設定/ウェルカムにある「ブログを見る」リンクを探す
    sels = [
        'a:has-text("ブログを見る")',
        'a[target="_blank"][href*="livedoor.blog"]',
        'a[href^="https://blog.livedoor.com/"]',
        'a[href^="https://blog.livedoor.jp/"]',
    ]
    for sel in sels:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                href = await loc.get_attribute("href")
                if href:
                    # 相対URLで返るケースに備えて絶対化
                    abs_href = await page.evaluate("href => new URL(href, location.href).href", href)
                    return abs_href
        except Exception:
            pass
    return None


async def recover_atompub_key(page, livedoor_id: str | None, nickname: str, email: str, password: str, site,
                              desired_blog_id: str | None = None) -> dict:
    """
    - Livedoorブログの作成 → AtomPub APIキーを発行・取得
    - CAPTCHA が無ければ通常送信で進む
    - CAPTCHA が出た場合は、サインアップ時同様の「ツールで手動入力」フローに切替（なければFSフォールバック）
    - 送信後に create に留まった場合は、規約同意チェックと blog_id 指定の再送も試す
    - DBには保存しない（呼び出し元で保存）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("[LD-Recover] args: livedoor_id=%s desired_blog_id=%s email=%s", livedoor_id, desired_blog_id, email)

    # 失敗時保存用ヘルパ
    async def _dump_error(prefix: str):
        html = await page.content()
        error_html = f"/tmp/{prefix}_{timestamp}.html"
        error_png = f"/tmp/{prefix}_{timestamp}.png"
        Path(error_html).write_text(html, encoding="utf-8")
        try:
            await page.screenshot(path=error_png, full_page=True)
        except Exception:
            pass
        return error_html, error_png

    try:
        # 1) ブログ作成ページへ
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        await _save_shot(page, "ld_create_landing")

        # 中間導線の踏破（同意・開始ボタンなど）
        try:
            logger.info("[LD-Recover] create到達: url=%s title=%s", page.url, (await page.title()))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            await page.screenshot(path=f"/tmp/ld_create_landing_{ts}.png", full_page=True)
            Path(f"/tmp/ld_create_landing_{ts}.html").write_text(await page.content(), encoding="utf-8")
            logger.info("[LD-Recover] dump: /tmp/ld_create_landing_%s.png /tmp/ld_create_landing_%s.html", ts, ts)
        except Exception:
            pass

        interstitial_sels = [
            'button:has-text("同意して進む")',
            'button:has-text("同意")',
            'button:has-text("許可")',
            'a:has-text("無料でブログを始める")',
            'a:has-text("ブログの作成を開始")',
            'a:has-text("新しくブログを作成")',
            'a:has-text("ブログを始める")',
            'button:has-text("ブログの作成")',
            'a.button:has-text("ブログを作成")',
        ]
        for sel in interstitial_sels:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    logger.info("[LD-Recover] 中間導線をクリック: %s", sel)
                    await _wait_enabled_and_click(page, loc, timeout=7000, label_for_log=f"interstitial {sel}")
                    try:
                        await page.wait_for_load_state("load", timeout=8000)
                    except Exception:
                        pass
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    break
            except Exception:
                continue

        # メール認証未完了の早期検知
        if "need_email_auth" in page.url:
            logger.warning("[LD-Recover] email auth required before blog creation: %s", page.url)
            return {
                "success": False,
                "error": "email_auth_required",
                "need_email_auth": True,
                "where": page.url,
            }
        # 2) 公開URLのサブドメイン（#sub）を「ユーザーID（desired_blog_id）」で先に指定
        #    ※ これを『最初の送信前』にやるのがポイント
        try:
            if await _has_blog_id_input(page) and desired_blog_id:
                await _try_set_desired_blog_id(page, desired_blog_id)
        except Exception:
            pass

        # 3) ブログタイトルを設定し送信
        try:
            desired_title = _craft_blog_title(site)
        except Exception:
            desired_title = "日々ブログ"

        logger.info("[LD-Recover] タイトル設定＆送信開始")
        ok_submit = await _set_title_and_submit(page, desired_title)
        if not ok_submit:
            err_html, err_png = await _dump_error("ld_create_ui_notfound")
            return {
                "success": False,
                "error": "ブログ作成UIが見つからない/クリック不可（DOM/iframe変更の可能性）",
                "html_path": err_html,
                "png_path": err_png,
            }

        # 4) 一発成功判定
        success, blog_id_from_url = await _wait_success_after_submit(page)

        # 5) まだ create に留まる → まず CAPTCHA を最優先で処理
        if not success:
            await _save_shot(page, "ld_create_after_submit_failed")
            await _log_inline_errors(page)

            # CAPTCHA 検出
            is_cap, cap_path = await _detect_create_captcha(page)
            if is_cap:
                # 画像更新ボタンがあれば一度更新（読みやすくするため・任意）
                # CAPTCHA 検出後の「画像更新」
                try:
                    await page.locator('#captchaImageA, a:has-text("画像を更新"), button:has-text("画像を更新")').first.click(timeout=1000)
                    await asyncio.sleep(0.8)
                    is_cap, cap_path = await _detect_create_captcha(page)
                except Exception:
                    pass

                # サインアップツール or FS 監視で人力回答を取得→入力・送信
                # livedoor_id が未指定なら URL サブドメインと同一の desired_blog_id を採用
                lid = livedoor_id or desired_blog_id
                ok_cap = await _human_tool_captcha_flow(
                    page,
                    cap_path or "",
                    livedoor_id=lid,  # ← ユーザーIDを渡す（メールではない）
                    password=password,
                )
                if ok_cap:
                    success, blog_id_from_url = await _wait_success_after_submit(page)
                    if success:
                        logger.info("[LD-Recover] /welcome へ遷移（captcha solved）")
                else:
                    logger.warning("[LD-Recover] CAPTCHA 処理に失敗（人力回答未取得 or 入力不可）")

        # 6) それでもダメなら：規約同意→再送、blog_id 候補→再送
        if not success:
            terms_changed = await _maybe_accept_terms(page)

            # 規約同意を付けた場合は、一度は再送
            if terms_changed and not success:
                if await _set_title_and_submit(page, desired_title):
                    success, blog_id_from_url = await _wait_success_after_submit(page)
                    if success:
                        logger.info("[LD-Recover] /welcome へ遷移（terms accepted）")

            # blog_id 入力欄があれば候補で再送（ユーザーIDが空の場合のみ自動候補へ）
            if not success:
                has_id_box = await _has_blog_id_input(page)
                if has_id_box:
                    if desired_blog_id:
                        candidates = [desired_blog_id]
                    else:
                        base = _slugify_ascii(getattr(site, "name", None) or getattr(site, "url", None) or "blog")
                        candidates = [base] + [f"{base}-{i}" for i in range(1, 8)]
                    for cand in candidates:
                        try:
                            if await _try_set_desired_blog_id(page, cand):
                                logger.info(f"[LD-Recover] blog_id 必須/衝突の可能性 → 候補で再送信: {cand}")
                                if not await _set_title_and_submit(page, desired_title):
                                    continue
                                success, blog_id_from_url = await _wait_success_after_submit(page)
                                if success:
                                    logger.info(f"[LD-Recover] /welcome へ遷移（blog_id={cand}）")
                                    break
                        except Exception:
                            continue

        if not success:
            # ここまで来たら失敗としてダンプ
            await _save_shot(page, "ld_create_still_failed")
            err_html, err_png = await _dump_error("ld_atompub_create_fail")
            logger.error("[LD-Recover] ブログ作成に失敗（createに留まる）")
            return {
                "success": False,
                "error": "blog create failed",
                "html_path": err_html,
                "png_path": err_png
            }

        # 7) （任意）welcome にある導線があれば押す
        try:
            fr, sel = await _find_in_any_frame(page, [
                'a:has-text("最初のブログを書く")',
                'a.button:has-text("はじめての投稿")',
            ], timeout_ms=2500)
            if fr and sel:
                try:
                    await _wait_enabled_and_click(fr, fr.locator(sel).first, timeout=3000, label_for_log="welcome-next")
                    logger.info("[LD-Recover] 『最初のブログを書く』をクリック（任意）")
                except Exception:
                    pass
        except Exception:
            pass

        # 8) blog_id 決定：まずは URL から、無ければ /member で探す
        blog_id = blog_id_from_url
        if blog_id:
            logger.info(f"[LD-Recover] ブログID（URL由来）: {blog_id}")
        else:
            # /member に遷移して抽出（従来ルート）
            await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            blog_settings_selectors = [
                'a[title="ブログ設定"]',
                'a:has-text("ブログ設定")',
                'a[href^="/blog/"][href$="/config/"]',
                'a[href*="/config/"]'
            ]

            href = None
            for sel in blog_settings_selectors:
                try:
                    loc = page.locator(sel).first
                    if await loc.count() > 0:
                        try:
                            await loc.wait_for(state="visible", timeout=8000)
                        except Exception:
                            pass
                        href = await loc.get_attribute("href")
                        if href:
                            break
                except Exception:
                    continue

            if not href:
                fr, sel = await _find_in_any_frame(page, blog_settings_selectors, timeout_ms=12000)
                if fr:
                    loc = fr.locator(sel).first
                    try:
                        await loc.wait_for(state="visible", timeout=6000)
                    except Exception:
                        pass
                    href = await loc.get_attribute("href")

            if href:
                try:
                    parts = href.split("/")
                    blog_id = parts[2] if len(parts) > 2 else None
                except Exception:
                    blog_id = None

            if not blog_id:
                cur_url = page.url
                blog_id = cur_url.split("/blog/")[1].split("/")[0] if "/blog/" in cur_url else None

            if not blog_id:
                err_html, err_png = await _dump_error("ld_atompub_member_fail")
                return {
                    "success": False,
                    "error": "member page missing blog link",
                    "html_path": err_html,
                    "png_path": err_png
                }

        # 9) 設定ページ → AtomPub 発行ページへ（blog_id を直接使って遷移）
        config_url = f"https://livedoor.blogcms.jp/blog/{blog_id}/config/"
        await page.goto(config_url, wait_until="load")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        api_nav_selectors = [
            'a.configIdxApi[title="API Keyの発行・確認"]',
            'a[title*="API Key"]',
            'a:has-text("API Key")',
            'a:has-text("API Keyの発行")',
            'a[href*="/api"]',
            'a:has-text("AtomPub")',
        ]
        api_link = None
        for sel in api_nav_selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    api_link = loc
                    break
            except Exception:
                continue
        if api_link is None:
            fr, sel = await _find_in_any_frame(page, api_nav_selectors, timeout_ms=8000)
            if fr:
                api_link = fr.locator(sel).first

        if api_link is None:
            err_html, err_png = await _dump_error("ld_atompub_nav_fail")
            logger.error("[LD-Recover] AtomPub設定ページへのリンクが見つからない")
            return {
                "success": False,
                "error": "api nav link not found",
                "html_path": err_html,
                "png_path": err_png
            }

        await _wait_enabled_and_click(page, api_link, timeout=8000, label_for_log="api-nav")
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass

        logger.info(f"[LD-Recover] AtomPub設定ページに遷移: {page.url}")

        if "member" in page.url:
            err_html, err_png = await _dump_error("ld_atompub_redirect_fail")
            logger.error(f"[LD-Recover] AtomPubページが開けず /member にリダイレクト: {page.url}")
            return {
                "success": False,
                "error": "redirected to member",
                "html_path": err_html,
                "png_path": err_png
            }

        # 10) スクショ
        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        try:
            await page.screenshot(path=success_png, full_page=True)
        except Exception:
            try:
                await page.screenshot(path=success_png)
            except Exception:
                pass
        logger.info(f"[LD-Recover] AtomPubページのスクリーンショット保存: {success_png}")

        # 11) APIキー発行
        await page.wait_for_selector('input#apiKeyIssue', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue")
        logger.info("[LD-Recover] 『発行する』をクリック")

        await page.wait_for_selector('button:has-text("実行")', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm")
        logger.info("[LD-Recover] モーダルの『実行』をクリック")

        # 12) 取得（valueはJSで後から入るため、非空になるまで待つ & 1回だけ再発行リトライ）
        async def _read_endpoint_and_key():
            endpoint_selectors = [
                'input.input-xxlarge[readonly]',
                'input[readonly][name*="endpoint"]',
                'input[readonly][id*="endpoint"]',
            ]
            endpoint_val = ""
            for sel in endpoint_selectors:
                try:
                    await page.wait_for_selector(sel, timeout=8000)
                    endpoint_val = await page.locator(sel).first.input_value()
                    if endpoint_val:
                        break
                except Exception:
                    continue

            await page.wait_for_selector('input#apiKey', timeout=15000)
            for _ in range(30):  # 30 * 0.5s = 15s
                key_val = (await page.locator('input#apiKey').input_value()).strip()
                if key_val:
                    return endpoint_val, key_val
                await asyncio.sleep(0.5)
            return endpoint_val, ""

        endpoint, api_key = await _read_endpoint_and_key()

        if not api_key:
            logger.warning("[LD-Recover] API Keyが空。ページを再読み込みして再発行をリトライ")
            await page.reload(wait_until="load")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            await page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue-retry")
            await page.wait_for_selector('button:has-text("実行")', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm-retry")
            endpoint, api_key = await _read_endpoint_and_key()

        if not api_key:
            err_html, err_png = await _dump_error("ld_atompub_no_key")
            logger.error(f"[LD-Recover] API Keyが取得できませんでした。証跡: {err_html}, {err_png}")
            return {
                "success": False,
                "error": "api key empty",
                "html_path": err_html,
                "png_path": err_png
            }

        logger.info(f"[LD-Recover] ✅ AtomPub endpoint: {endpoint}")
        logger.info(f"[LD-Recover] ✅ AtomPub key: {api_key[:8]}...")

        # 13) 公開URL抽出（推測せずリンクから）
        public_url = await _extract_public_url(page)
        if not public_url:
            # AtomPubページに無い場合は config 直下へ戻って再探索
            try:
                await page.goto(f"https://livedoor.blogcms.jp/blog/{blog_id}/config/", wait_until="load")
                try:
                    await page.wait_for_load_state("networkidle", timeout=6000)
                except Exception:
                    pass
                public_url = await _extract_public_url(page)
            except Exception:
                pass

        # 14) 返却（DB保存は呼び出し元が担当）
        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "blog_title": desired_title,  # 任意
            "public_url": public_url      # ← 追加：実際の公開URL
        }

    except Exception as e:
        err_html, err_png = await _dump_error("ld_atompub_fail")
        logger.error("[LD-Recover] AtomPub処理エラー", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "html_path": err_html,
            "png_path": err_png
        }
