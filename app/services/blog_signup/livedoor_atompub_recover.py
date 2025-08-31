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
BUILD_TAG = "2025-08-29 livedoor-create-minimal(title+submit-only)"
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
    サイト名から「1語」を安定に選ぶ。
    戻り値: (seed, is_jp)  /  抽出できなければ (None, False)
    """
    name = (getattr(site, "name", "") or "").strip()
    # salt は id+name のみ（URLは含めない）
    salt = f"{getattr(site, 'id', '')}-{name}"

    name_toks = _name_tokens(name)

    # 日本語と英語で候補を分ける
    jp_cands = [t for t in name_toks if _has_cjk(t) and t not in STOPWORDS_JP]
    en_cands = [t for t in name_toks if not _has_cjk(t)]
    en_cands = [t for t in en_cands if t.lower() not in STOPWORDS_EN]

    # 長さフィルタ（1文字や長すぎは除外）
    jp_cands = [t for t in jp_cands if 2 <= len(t) <= 12]
    en_cands = [t for t in en_cands if 2 <= len(t) <= 15]

    # 同一サイトでは安定して同じ語を選ぶ（塩＝site.id+name）
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
    2) site.name の語からヒューリスティック（URLは参照しない）
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

    # 2) ヒューリスティック（サイト名のみ）
    name = (getattr(site, "name", "") or "")
    txt = name.lower()

    JP = [
        ("ピラティス", ("pilates", "ピラティス", "yoga", "体幹", "姿勢", "fitness", "stretch")),
        ("留学", ("studyabroad", "abroad", "留学", "ielts", "toefl", "海外", "study")),
        ("旅行", ("travel", "trip", "観光", "hotel", "onsen", "温泉", "tour")),
        ("美容", ("beauty", "esthetic", "skin", "hair", "美容", "コスメ", "メイク")),
        ("ビジネス", ("business", "marketing", "sales", "seo", "経営", "起業", "副業")),
    ]
    for label, keys in JP:
        if any(k in txt for k in keys):
            return label, True

    EN = [
        ("Pilates", ("pilates", "yoga", "fitness", "posture", "stretch")),
        ("Study Abroad", ("studyabroad", "abroad", "study", "ielts", "toefl")),
        ("Travel", ("travel", "trip", "hotel", "onsen", "tour")),
        ("Beauty", ("beauty", "esthetic", "skin", "hair", "cosme", "makeup")),
        ("Business", ("business", "marketing", "sales", "seo", "startup")),
    ]
    for label, keys in EN:
        if any(k in txt for k in keys):
            return label, False

    # どれにも該当しなければ汎用
    return ("日々", _has_cjk(name))


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
    仕様（ご指定反映）：
      - 生成結果は日本語ベース
      - サイト名/URLから抽出したキーワードやジャンル語を使って“ブログ風”に
      - 「日々のブログ」にはしない（明示的に禁止）
      - 元サイト名/ドメインに似すぎない
      - 同一サイトでは決定論的に安定
    """
    site_name = (getattr(site, "name", "") or "").strip()
    site_url = (getattr(site, "url", "") or "").strip()
    salt = f"{getattr(site, 'id', '')}-{site_name}-{site_url}"

    # まずはサイトから1語シードを取る（日本語があれば優先）
    seed, seed_is_jp = _keyword_seed_from_site(site)
    if not seed:
        # ジャンル推定語（JPなら優先）
        topic, is_jp = _guess_genre(site)
        seed = topic if _has_cjk(topic) else "暮らし"  # デフォルトは「暮らし」
        seed_is_jp = True

    # ブログ風テンプレ（“ブログ”を含むパターン中心＋バリエーション）
    base = seed.strip()
    # 「日々のブログ」は禁止語として明示除外
    banned_exact = {"日々のブログ", "ひびのブログ"}
    candidates = [
        f"{base}ブログ",
        f"{base}のブログ",
        f"{base}ブログ記録",
        f"{base}の記録ブログ",
        f"{base}のメモブログ",
        f"{base}のノート",
        f"{base}ログ",
        f"{base}手帖",
    ]

    # 許容判定
    def acceptable(title: str) -> bool:
        if not title or not title.strip():
            return False
        if title in banned_exact:
            return False
        if _too_similar_to_site(title, site):
            return False
        # 日本語らしさ：少なくとも1文字はCJK
        if not _has_cjk(title):
            return False
        return True

    # saltで開始位置を決め、順回しで最初に通ったものを採用
    start = _deterministic_index(salt, len(candidates))
    for i in range(len(candidates)):
        t = candidates[(start + i) % len(candidates)]
        if acceptable(t):
            return t[:48]

    # 最終フォールバック（禁止の「日々のブログ」は含めない）
    fallbacks = [f"{base}ブログ", f"{base}ログ", f"{base}手帖", "こつこつブログ"]
    return fallbacks[_deterministic_index(salt, len(fallbacks))][:48]



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
    ブログタイトル入力 → 『ブログを作成する』クリック だけ。
    それ以外のUI操作は行わない。
    """
    title_selectors = ['#blogTitle', 'input[name="title"]']
    button_selectors = [
        'input[type="submit"][value="ブログを作成する"]',
        'button[type="submit"]',
        'input[type="submit"]',
    ]

    # タイトル入力
    title_loc = None
    for sel in title_selectors:
        try:
            await page.wait_for_selector(sel, state="visible", timeout=20000)
            cand = page.locator(sel).first
            if await cand.count() > 0:
                title_loc = cand
                break
        except Exception:
            continue
    if not title_loc:
        logger.warning("[LD-Recover] タイトル入力欄が見つかりません")
        return False

    try:
        try:
            await title_loc.fill("")
        except Exception:
            try:
                await title_loc.click()
                await title_loc.press("Control+A")
                await title_loc.press("Delete")
            except Exception:
                pass
        await title_loc.fill(desired_title)
        logger.info("[LD-Recover] ブログタイトルを設定: %s", desired_title)
    except Exception:
        logger.warning("[LD-Recover] タイトル入力に失敗", exc_info=True)
        return False

    # 作成ボタン
    btn = None
    for sel in button_selectors:
        try:
            cand = page.locator(sel).first
            if await cand.count() > 0:
                btn = cand
                break
        except Exception:
            continue
    if not btn:
        logger.warning("[LD-Recover] 『ブログを作成する』ボタンが見つかりません")
        return False

    # クリック（遷移が発生しないUIでも1回だけ押す）
    try:
        async with page.expect_navigation(wait_until="load", timeout=15000):
            await btn.click()
        logger.info("[LD-Recover] 『ブログを作成する』をクリック")
    except Exception:
        # 遷移イベントが取れなくても、1回だけフォールバッククリック
        try:
            await btn.click(timeout=5000)
            logger.info("[LD-Recover] 『ブログを作成する』をクリック（fallback）")
        except Exception:
            logger.warning("[LD-Recover] 作成ボタンクリックに失敗", exc_info=True)
            return False

    # 追加の安定待ち（軽く）
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("[LD-Recover] args: livedoor_id=%s desired_blog_id=%s email=%s", livedoor_id, desired_blog_id, email)

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
        logger.info("[LD-Recover] create到達: url=%s title=%s", page.url, (await page.title()))

        # 2) タイトル生成 → 送信（それ以外は何もしない）
        try:
            desired_title = _craft_blog_title(site)
        except Exception:
            desired_title = "こつこつブログ"  # 「日々のブログ」は使わない安全フォールバック

        ok_submit = await _set_title_and_submit(page, desired_title)
        if not ok_submit:
            err_html, err_png = await _dump_error("ld_create_ui_notfound")
            return {"success": False, "error": "タイトル/送信UIが見つからない", "html_path": err_html, "png_path": err_png}

        # 3) 成功判定
        success, blog_id_from_url = await _wait_success_after_submit(page)
        if not success:
            await _save_shot(page, "ld_create_after_submit_failed_minimal")
            await _log_inline_errors(page)
            err_html, err_png = await _dump_error("ld_atompub_create_fail_minimal")
            logger.error("[LD-Recover] ブログ作成に失敗（createに留まる）")
            return {"success": False, "error": "blog create failed", "html_path": err_html, "png_path": err_png}

        # （以下は従来どおり：blog_id抽出→設定→APIキー取得）
        blog_id = blog_id_from_url
        if not blog_id:
            await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            # blog_idをメニューから推定
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
                        href = await loc.get_attribute("href"); 
                        if href: break
                except Exception:
                    continue
            if href:
                try:
                    parts = href.split("/")
                    blog_id = parts[2] if len(parts) > 2 else None
                except Exception:
                    blog_id = None
            if not blog_id and "/blog/" in page.url:
                blog_id = page.url.split("/blog/")[1].split("/")[0]
            if not blog_id:
                err_html, err_png = await _dump_error("ld_atompub_member_fail")
                return {"success": False, "error": "member page missing blog link", "html_path": err_html, "png_path": err_png}

        # 設定→APIキー
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
                    api_link = loc; break
            except Exception:
                continue
        if not api_link:
            fr, sel = await _find_in_any_frame(page, api_nav_selectors, timeout_ms=8000)
            if fr: api_link = fr.locator(sel).first
        if not api_link:
            err_html, err_png = await _dump_error("ld_atompub_nav_fail")
            return {"success": False, "error": "api nav link not found", "html_path": err_html, "png_path": err_png}

        await _wait_enabled_and_click(page, api_link, timeout=8000, label_for_log="api-nav")
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass
        if "member" in page.url:
            err_html, err_png = await _dump_error("ld_atompub_redirect_fail")
            return {"success": False, "error": "redirected to member", "html_path": err_html, "png_path": err_png}

        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        try:
            await page.screenshot(path=success_png, full_page=True)
        except Exception:
            try: await page.screenshot(path=success_png)
            except Exception: pass

        # 発行
        await page.wait_for_selector('input#apiKeyIssue', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue")
        await page.wait_for_selector('button:has-text("実行")', timeout=12000)
        await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm")

        async def _read_endpoint_and_key():
            endpoint_val = ""
            for sel in ['input.input-xxlarge[readonly]','input[readonly][name*="endpoint"]','input[readonly][id*="endpoint"]']:
                try:
                    await page.wait_for_selector(sel, timeout=8000)
                    endpoint_val = await page.locator(sel).first.input_value()
                    if endpoint_val: break
                except Exception:
                    continue
            await page.wait_for_selector('input#apiKey', timeout=15000)
            for _ in range(30):
                key_val = (await page.locator('input#apiKey').input_value()).strip()
                if key_val: return endpoint_val, key_val
                await asyncio.sleep(0.5)
            return endpoint_val, ""

        endpoint, api_key = await _read_endpoint_and_key()
        if not api_key:
            await page.reload(wait_until="load")
            try: await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception: pass
            await page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('input#apiKeyIssue').first, timeout=6000, label_for_log="api-issue-retry")
            await page.wait_for_selector('button:has-text("実行")', timeout=15000)
            await _wait_enabled_and_click(page, page.locator('button:has-text("実行")').first, timeout=6000, label_for_log="api-issue-confirm-retry")
            endpoint, api_key = await _read_endpoint_and_key()
        if not api_key:
            err_html, err_png = await _dump_error("ld_atompub_no_key")
            return {"success": False, "error": "api key empty", "html_path": err_html, "png_path": err_png}

        public_url = await _extract_public_url(page)
        if not public_url:
            try:
                await page.goto(f"https://livedoor.blogcms.jp/blog/{blog_id}/config/", wait_until="load")
                try: await page.wait_for_load_state("networkidle", timeout=6000)
                except Exception: pass
                public_url = await _extract_public_url(page)
            except Exception:
                pass

        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "blog_title": desired_title,
            "public_url": public_url
        }

    except Exception as e:
        err_html, err_png = await _dump_error("ld_atompub_fail")
        logger.error("[LD-Recover] AtomPub処理エラー", exc_info=True)
        return {"success": False, "error": str(e), "html_path": err_html, "png_path": err_png}

