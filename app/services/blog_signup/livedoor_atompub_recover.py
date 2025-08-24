import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re as _re
from typing import Optional, Dict, List, Tuple

from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 汎用ヘルパ
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


def _domain_tokens(url: str) -> List[str]:
    """ドメインを単語に分割（tld等は除外）"""
    try:
        netloc = urlparse(url or "").netloc.lower()
    except Exception:
        netloc = ""
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    parts = [p for p in netloc.split(".") if p and p not in ("www", "com", "jp", "net", "org", "co")]
    words: List[str] = []
    for p in parts:
        words.extend([w for w in p.replace("_", "-").split("-") if w])
    return words


STOPWORDS_JP = {
    "株式会社", "有限会社", "合同会社", "公式", "オフィシャル", "ブログ", "サイト", "ホームページ",
    "ショップ", "ストア", "サービス", "工房", "教室", "情報", "案内", "チャンネル", "通信", "マガジン"
}
STOPWORDS_EN = {
    "inc", "ltd", "llc", "official", "blog", "site", "homepage", "shop", "store",
    "service", "studio", "channel", "magazine", "info", "news"
}


def _name_tokens(name: str) -> List[str]:
    """サイト名を雑にトークン化（日本語/英語混在対応・記号で分割）"""
    if not name:
        return []
    parts = _re.split(r"[\s\u3000\-/＿_・|｜／]+", str(name))
    toks: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 記号を除去しすぎない程度に掃除
        p = _re.sub(r"[^\w\u3040-\u30FF\u3400-\u9FFFー]+", "", p)
        if p:
            toks.append(p)
    return toks


def _keyword_seed_from_site(site) -> Tuple[Optional[str], bool]:
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

    def _pick(stable_list: List[str]) -> Optional[str]:
        if not stable_list:
            return None
        idx = _deterministic_index(salt, len(stable_list))
        return stable_list[idx]

    seed = _pick(jp_cands) or _pick(en_cands)
    if seed:
        return seed, _has_cjk(seed)
    return None, False


def _guess_genre(site) -> Tuple[str, bool]:
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


def _templates_jp(topic: str) -> List[str]:
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


def _templates_en(topic: str) -> List[str]:
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
        # “ブログっぽさ”の担保（語に「ブログ」が含まれることを最低条件に）
        if "ブログ" not in title:
            return False
        return True

    # 4) salt で開始位置を決め、順送りで最初に通ったもの
    idx = _deterministic_index(salt, len(cands))
    for i in range(len(cands)):
        title = cands[(idx + i) % len(cands)]
        if acceptable(title):
            return title[:48]

    # 5) 最終フォールバック（必ず日本語＆ブログ語尾）
    fallback = [f"{base_word}ブログ", f"{base_word}のブログ", "日々ブログ", "小さなブログ記録"]
    return fallback[_deterministic_index(salt, len(fallback))][:48]


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


# ─────────────────────────────────────────────
# Playwright 補助
# ─────────────────────────────────────────────
async def _try_set_desired_blog_id(page, desired: str) -> bool:
    """
    ブログ作成画面で希望 blog_id を入力する。
    画面ごとの違いに備えて複数セレクタを順に試す。
    成功/入力欄が見つからない場合は True、明確な失敗で False を返す。
    """
    selectors = [
        '#blogId',
        'input[name="blog_id"]',
        'input[name="livedoor_blog_id"]',
        'input[name="blogId"]',
        'input#livedoor_blog_id',
    ]
    try:
        for sel in selectors:
            try:
                if await page.locator(sel).count() > 0:
                    await page.fill(sel, desired)
                    return True
            except Exception:
                continue
        # 入力欄が見つからなくても致命ではない（サイト側が自動採番の可能性）
        return True
    except Exception:
        return False


async def _maybe_accept_agreement(page) -> None:
    """規約チェック・同意画面などがあれば best-effort で突破"""
    # チェックボックス
    for sel in [
        'input[type="checkbox"][name*="agree"]',
        'input[type="checkbox"][id*="agree"]',
    ]:
        try:
            await page.check(sel, timeout=2000)
            logger.info("[LD-Recover] checked agreement: %s", sel)
            break
        except Exception:
            pass

    # 同意/次へ/OK
    for sel in [
        'input[type="submit"][value*="同意"]',
        'input[type="submit"][value*="進む"]',
        'button:has-text("同意")',
        'button:has-text("次へ")',
        'a:has-text("同意")',
        'a:has-text("次へ")',
    ]:
        try:
            await page.click(sel, timeout=2000)
            logger.info("[LD-Recover] clicked agreement/next: %s", sel)
            await page.wait_for_load_state("networkidle", timeout=8000)
            break
        except Exception:
            pass


async def _on_blog_create_page(page) -> None:
    if "blog/create" not in page.url:
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")


async def _click_create_button(page) -> bool:
    """『ブログを作成する』系のボタンを柔軟にクリック"""
    candidates = [
        'input[type="submit"][value="ブログを作成する"]',
        'input[type="submit"][value*="ブログを作成"]',
        'input[type="submit"][value*="ブログ作成"]',
        'input[type="submit"][value*="作成"]',
        'input[type="submit"][value*="登録"]',
        '#commit-button',
        'button[type="submit"]',
        'button:has-text("作成")',
        'button:has-text("登録")',
    ]
    for sel in candidates:
        try:
            await page.wait_for_selector(sel, timeout=6000)
            await page.click(sel, timeout=6000)
            logger.info("[LD-Recover] クリック: %s", sel)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            return True
        except Exception:
            continue
    return False


async def _success_after_create(page) -> bool:
    """作成成功の判定（URL/導線/ダッシュボード遷移など複数の基準）"""
    # 1) /welcome
    try:
        await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=8000)
        logger.info("[LD-Recover] /welcome への遷移を確認")
        return True
    except Exception:
        pass

    # 2) member ダッシュボード系
    if "livedoor.blogcms.jp/member/" in page.url:
        logger.info("[LD-Recover] member 画面に遷移: %s", page.url)
        return True

    # 3) welcome ボタンなど
    try:
        await page.wait_for_selector(
            'a:has-text("最初のブログを書く"), a.button:has-text("はじめての投稿")',
            timeout=4000
        )
        logger.info("[LD-Recover] welcome 導線（ボタン）の出現を確認")
        return True
    except Exception:
        pass

    return False


async def _find_member_blog_config_link(page) -> Optional[str]:
    """
    /member 画面から blog 設定リンクの href を探す。
    失敗時は None。
    """
    # 直接の title 属性
    try:
        if await page.locator('a[title="ブログ設定"]').count() > 0:
            href = await page.get_attribute('a[title="ブログ設定"]', 'href')
            if href:
                return href
    except Exception:
        pass

    # href パターン
    try:
        loc = page.locator('a[href*="/member/blog/"][href$="/config/"]')
        if await loc.count() > 0:
            href = await loc.first.get_attribute('href')
            if href:
                return href
    except Exception:
        pass

    # リンクテキスト（日本語揺れに弱いが一応）
    for txt in ["ブログ設定", "設定", "Config", "Settings"]:
        try:
            loc = page.locator(f'a:has-text("{txt}")')
            if await loc.count() > 0:
                href = await loc.first.get_attribute('href')
                if href and "/member/blog/" in href and "/config/" in href:
                    return href
        except Exception:
            pass

    return None


async def _open_api_settings(page, blog_id: Optional[str]) -> bool:
    """
    API/AtomPub 設定ページへ遷移。リンクテキスト/URLが変わっても複数パターンで辿る
    """
    # 1) blog_id があれば直接 URL を試す
    direct_urls = []
    if blog_id:
        direct_urls += [
            f"https://livedoor.blogcms.jp/member/blog/{blog_id}/api",
            f"https://livedoor.blogcms.jp/member/blog/{blog_id}/atompub",
        ]
    # 2) 旧来の汎用候補
    direct_urls += [
        "https://livedoor.blogcms.jp/member/api/atompub",
        "https://livedoor.blogcms.jp/member/api",
    ]
    for u in direct_urls:
        try:
            await page.goto(u, wait_until="load", timeout=10000)
            if "api" in page.url or "atom" in page.url:
                logger.info("[LD-Recover] API設定ページ（直接URL）: %s", page.url)
                return True
        except Exception:
            continue

    # 3) UI ナビから当てる
    link_texts = ["API", "AtomPub", "API設定", "AtomPub設定", "API Key", "キー発行"]
    for txt in link_texts:
        try:
            await page.click(f'a:has-text("{txt}")', timeout=5000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            if "api" in page.url or "atom" in page.url:
                logger.info("[LD-Recover] API設定ページ（UI経由）: %s", page.url)
                return True
        except Exception:
            continue

    # 4) CSS クラスから（従来）
    try:
        await page.wait_for_selector('a.configIdxApi[title="API Keyの発行・確認"]', timeout=5000)
        await page.click('a.configIdxApi[title="API Keyの発行・確認"]', timeout=5000)
        await page.wait_for_load_state("load")
        if "api" in page.url or "atom" in page.url:
            logger.info("[LD-Recover] API設定ページ（configIdxApi）: %s", page.url)
            return True
    except Exception:
        pass

    return False


async def _extract_endpoint_and_key(page) -> Tuple[Optional[str], Optional[str]]:
    """
    ページから AtomPub エンドポイントと API Key を読み取る。
    - 既に発行済みの場合はクリック前に読めることもあるため、先に読む
    - 未発行なら『発行する』→『実行』で発行してから読む
    """
    async def _read() -> Tuple[Optional[str], Optional[str]]:
        endpoint = None
        key = None

        # endpoint 候補
        endpoint_sel = [
            'input.input-xxlarge[readonly]',
            'input#atompubUrl',
            'input[name="atompub_url"]',
            'input[name="endpoint"]',
            '#endpoint',
            '#atompub-url',
        ]
        for sel in endpoint_sel:
            try:
                if await page.locator(sel).count() > 0:
                    endpoint = (await page.locator(sel).first.input_value()).strip()
                    if endpoint:
                        break
            except Exception:
                continue

        # key 候補
        key_sel = [
            'input#apiKey',
            'input[name="api_key"]',
            '#api_key',
            '.api-key',
        ]
        for sel in key_sel:
            try:
                if await page.locator(sel).count() > 0:
                    key = (await page.locator(sel).first.input_value()).strip()
                    if key:
                        break
            except Exception:
                continue

        # どうしても取れない場合はページ全体からそれっぽいトークンを拾う（最後の手段）
        if not key:
            try:
                html = await page.content()
                m = _re.search(r"\b[A-Za-z0-9]{24,64}\b", html)
                if m:
                    key = m.group(0)
            except Exception:
                pass

        return endpoint, key

    # 既に値があるか先に読む
    endpoint, api_key = await _read()
    if api_key:
        return endpoint, api_key

    # 未発行なら『発行する』→『実行』
    try:
        if await page.locator('input#apiKeyIssue').count() > 0:
            await page.click('input#apiKeyIssue', timeout=15000)
            logger.info("[LD-Recover] 『発行する』をクリック")
            if await page.locator('button:has-text("実行")').count() > 0:
                await page.click('button:has-text("実行")', timeout=15000)
                logger.info("[LD-Recover] モーダルの『実行』をクリック")
    except Exception:
        logger.warning("[LD-Recover] APIキー発行ボタンのクリックに失敗（既に発行済みの可能性）", exc_info=True)

    # 値が入るまで一定時間ポーリング
    try:
        for _ in range(30):  # 15秒
            endpoint, api_key = await _read()
            if api_key:
                return endpoint, api_key
            await asyncio.sleep(0.5)
    except Exception:
        pass

    # リロード・再発行リトライ（1回）
    try:
        await page.reload(wait_until="load")
        if await page.locator('input#apiKeyIssue').count() > 0:
            await page.click('input#apiKeyIssue', timeout=15000)
            if await page.locator('button:has-text("実行")').count() > 0:
                await page.click('button:has-text("実行")', timeout=15000)
        for _ in range(30):
            endpoint, api_key = await _read()
            if api_key:
                return endpoint, api_key
            await asyncio.sleep(0.5)
    except Exception:
        pass

    return endpoint, api_key


# ─────────────────────────────────────────────
# メイン：recover_atompub_key
# ─────────────────────────────────────────────
async def recover_atompub_key(
    page,
    nickname: str,
    email: str,
    password: str,
    site,
    desired_blog_id: Optional[str] = None
) -> Dict:
    """
    メール認証後の状態で呼ばれる前提。
    1) ブログ作成（ID重複時はエラーメッセージ検出しつつ最小限のフォールバック）
    2) member 画面から blog_id を把握
    3) API設定ページへ遷移
    4) APIキー抽出
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def _dump_error(prefix: str) -> Tuple[str, str]:
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
        await _on_blog_create_page(page)
        await _maybe_accept_agreement(page)

        # タイトルを設定
        desired_title = None
        try:
            desired_title = _craft_blog_title(site)
            await page.wait_for_selector('#blogTitle, input[name="title"]', timeout=15000)
            for sel in ['#blogTitle', 'input#blogTitle', 'input[name="title"]']:
                try:
                    if await page.locator(sel).count() > 0:
                        el = page.locator(sel)
                        try:
                            await el.fill("")
                        except Exception:
                            pass
                        try:
                            await el.press("Control+A")
                            await el.press("Delete")
                        except Exception:
                            pass
                        await el.fill(desired_title)
                        logger.info("[LD-Recover] ブログタイトルを設定: %s (%s)", desired_title, sel)
                        break
                except Exception:
                    continue
        except Exception:
            logger.warning("[LD-Recover] タイトル入力欄の操作に失敗（非致命）", exc_info=True)

        # 作成ボタン
        if not await _click_create_button(page):
            err_html, err_png = await _dump_error("ld_create_button_missing")
            return {
                "success": False,
                "error": "ブログ作成ボタンが見つかりません",
                "html_path": err_html,
                "png_path": err_png
            }

        # 成功判定
        success = await _success_after_create(page)
        if not success:
            # 同一ページに留まっている＝何らかのバリデーションで弾かれた可能性
            html_lower = (await page.content()).lower()
            dup_or_required = any(k in html_lower for k in [
                "使用できません", "既に使われています", "重複", "invalid", "already",
                "必須", "入力してください"
            ])

            if dup_or_required:
                # 最小限のフォールバック：blog_id 入力欄があるときだけ入れて再送信
                base = _slugify_ascii(getattr(site, "name", None) or getattr(site, "url", None) or "blog")
                candidates = [desired_blog_id] if desired_blog_id else []
                candidates += [base] + [f"{base}-{i}" for i in range(1, 6)]

                # blog_id 入力欄がある？
                has_id_box = False
                for sel in ['#blogId', 'input[name="blog_id"]', 'input[name="livedoor_blog_id"]', 'input[name="blogId"]', 'input#livedoor_blog_id']:
                    try:
                        if await page.locator(sel).count() > 0:
                            has_id_box = True
                            break
                    except Exception:
                        pass

                if has_id_box:
                    for cand in candidates:
                        if not cand:
                            continue
                        try:
                            if await _try_set_desired_blog_id(page, cand):
                                logger.info("[LD-Recover] blog_id 衝突/必須を検知 → 候補で再試行: %s", cand)
                                # もう一度ボタン
                                await _click_create_button(page)
                                success = await _success_after_create(page)
                                if success:
                                    logger.info("[LD-Recover] ブログ作成成功（blog_id=%s）", cand)
                                    break
                        except Exception:
                            continue

            if not success:
                err_html, err_png = await _dump_error("ld_atompub_create_fail")
                logger.error("[LD-Recover] ブログ作成に失敗（タイトルのみ or 自動採番不可）")
                return {
                    "success": False,
                    "error": "blog create failed",
                    "html_path": err_html,
                    "png_path": err_png
                }

        # welcome にボタンがあれば押す（任意）
        try:
            await page.click('a:has-text("最初のブログを書く"), a.button:has-text("はじめての投稿")', timeout=3000)
            logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")
        except Exception:
            pass

        # 2) /member に遷移して blog_id を抽出
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        blog_url = await _find_member_blog_config_link(page)
        if not blog_url:
            # 末端UIが変わっている可能性 → 現在 URL から推測
            # 例: https://livedoor.blogcms.jp/member/blog/{blog_id}/xxx
            m = _re.search(r"/member/blog/([^/]+)/", page.url)
            fallback_blog_id = m.group(1) if m else None
            if not fallback_blog_id:
                err_html, err_png = await _dump_error("ld_atompub_member_fail")
                return {
                    "success": False,
                    "error": "member page missing blog link",
                    "html_path": err_html,
                    "png_path": err_png
                }
            blog_id = fallback_blog_id
            config_url = f"https://livedoor.blogcms.jp/member/blog/{blog_id}/config/"
        else:
            blog_id = blog_url.split("/")[2] if "/blog/" in blog_url else None
            if not blog_id:
                # /member/blog/{id}/config/ 想定
                m = _re.search(r"/member/blog/([^/]+)/", blog_url)
                blog_id = m.group(1) if m else None
            config_url = f"https://livedoor.blogcms.jp{blog_url}" if blog_url.startswith("/") else blog_url

        logger.info("[LD-Recover] ブログIDを取得: %s", blog_id)

        await page.goto(config_url, wait_until="load")

        # 3) API設定ページへ
        ok = await _open_api_settings(page, blog_id)
        if not ok:
            err_html, err_png = await _dump_error("ld_atompub_open_api_fail")
            logger.error("[LD-Recover] API設定ページに到達できませんでした")
            return {
                "success": False,
                "error": "API設定ページに到達できませんでした",
                "html_path": err_html,
                "png_path": err_png
            }

        # 4) エンドポイント & APIキー抽出（必要なら発行）
        endpoint, api_key = await _extract_endpoint_and_key(page)

        if not api_key:
            err_html, err_png = await _dump_error("ld_atompub_no_key")
            logger.error("[LD-Recover] API Keyが取得できませんでした。証跡: %s, %s", err_html, err_png)
            return {
                "success": False,
                "error": "api key empty",
                "html_path": err_html,
                "png_path": err_png
            }

        if not endpoint:
            # 既定候補にフォールバック
            endpoint = "https://livedoor.blogcms.jp/atompub"

        # スクリーンショット（成功時の証跡）
        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        try:
            await page.screenshot(path=success_png, full_page=True)
            logger.info("[LD-Recover] AtomPubページのスクリーンショット保存: %s", success_png)
        except Exception:
            pass

        logger.info("[LD-Recover] ✅ AtomPub endpoint: %s", endpoint)
        logger.info("[LD-Recover] ✅ AtomPub key: %s...", api_key[:8])

        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "blog_title": desired_title
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
