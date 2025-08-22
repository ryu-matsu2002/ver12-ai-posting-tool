import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re as _re

from app import db
from app.models import ExternalBlogAccount
from app.enums import BlogType
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

# ファイル先頭の import 群の近くに追加
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

# ===== 抽出用ヘルパーを追加 =====

STOPWORDS_JP = {
    "株式会社","有限会社","合同会社","公式","オフィシャル","ブログ","サイト","ホームページ",
    "ショップ","ストア","サービス","工房","教室","情報","案内","チャンネル","通信","マガジン"
}
STOPWORDS_EN = {
    "inc","ltd","llc","official","blog","site","homepage","shop","store",
    "service","studio","channel","magazine","info","news"
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
    url  = (getattr(site, "url", "") or "").strip()
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
    url  = (getattr(site, "url", "")  or "")
    txt  = (name + " " + url).lower()
    toks = set(_domain_tokens(url))

    JP = [
        ("ピラティス", ("pilates","ピラティス","yoga","体幹","姿勢","fitness","stretch")),
        ("留学",       ("studyabroad","abroad","留学","ielts","toefl","海外","study")),
        ("旅行",       ("travel","trip","観光","hotel","onsen","温泉","tour")),
        ("美容",       ("beauty","esthetic","skin","hair","美容","コスメ","メイク")),
        ("ビジネス",   ("business","marketing","sales","seo","経営","起業","副業")),
    ]
    for label, keys in JP:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, True

    EN = [
        ("Pilates", ("pilates","yoga","fitness","posture","stretch")),
        ("Study Abroad", ("studyabroad","abroad","study","ielts","toefl")),
        ("Travel", ("travel","trip","hotel","onsen","tour")),
        ("Beauty", ("beauty","esthetic","skin","hair","cosme","makeup")),
        ("Business", ("business","marketing","sales","seo","startup")),
    ]
    for label, keys in EN:
        if any(k in txt for k in keys) or any(k in toks for k in keys):
            return label, False

    # どれにも該当しなければ汎用
    return ("日々", _has_cjk(name) or _has_cjk(url))

# ===============================
# ★ 追加：類似度判定（“似すぎ”ブロック）
# ===============================
def _too_similar_to_site(title: str, site) -> bool:
    """
    タイトルがサイト名/ドメイン由来語と似すぎなら True。
    - 正規化同士の完全一致
    - 片方がもう片方を包含
    - ドメイン語幹（tokens）が含まれる/含まれる
    """
    t = _norm(title)
    site_name = (getattr(site, "name", "") or "")
    site_url  = (getattr(site, "url", "")  or "")
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


# ===============================
# ★ 修正：日本語テンプレを“ブログ風”に寄せる
# ===============================
def _templates_jp(topic: str) -> list[str]:
    base = (topic or "").strip() or "日々"
    # “ブログ”を含む語尾を優先（上から順に試す）
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

# 英語テンプレは使わないが、残しておく（呼ばれない想定）
def _templates_en(topic: str) -> list[str]:
    base = topic.strip() or "Notes"
    return [f"{base} Blog"]  # ダミー（呼ばれない想定）


# ===============================
# ★ 追加：日本語ベース語の決定
# ===============================
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


# ===============================
# ★ 差し替え：必ず日本語 & 似すぎ禁止のブログ名にする
# ===============================
def _craft_blog_title(site) -> str:
    """
    仕様：
      - 生成結果は必ず“日本語”
      - 元サイト名やドメインに“似すぎない”
      - かならず“ブログっぽい”語尾（～ブログ 等）を含める
      - 同一サイトでは決定論的に安定
    """
    site_name = (getattr(site, "name", "") or "").strip()
    site_url  = (getattr(site, "url", "")  or "").strip()
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


async def recover_atompub_key(page, nickname: str, email: str, password: str, site,
                              desired_blog_id: str | None = None) -> dict:
    """
    - Livedoorブログの作成 → AtomPub APIキーを発行・取得
    - 原則は「ブログタイトル入力 → 作成ボタン」だけを操作し、blog_id には触れない
    - ただし、送信後にページ内エラー（重複/必須など）を検知した場合に限り、
      最小限のフォールバックとして blog_id を入力して再送信を試みる
    - DBには保存しない（呼び出し元で保存）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 失敗時保存用ヘルパ
    async def _dump_error(prefix: str):
        html = await page.content()
        error_html = f"/tmp/{prefix}_{timestamp}.html"
        error_png = f"/tmp/{prefix}_{timestamp}.png"
        Path(error_html).write_text(html, encoding="utf-8")
        await page.screenshot(path=error_png)
        return error_html, error_png

    try:
        # 1) ブログ作成ページへ
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")

        # ブログタイトルを“サイト名に関連する”名前に置き換える（ここだけ触る）
        desired_title = None
        try:
            desired_title = _craft_blog_title(site)
            # フィールドを待ってから、既定値を一度クリアして上書き
            await page.wait_for_selector('#blogTitle, input[name="title"]', timeout=10000)
            title_selectors = ['#blogTitle', 'input#blogTitle', 'input[name="title"]']
            for sel in title_selectors:
                if await page.locator(sel).count() > 0:
                    el = page.locator(sel)
                    # まず空に（既定値を消す）
                    try:
                        await el.fill("")
                    except Exception:
                        pass
                    # 念のため全選択→Delete でもクリア
                    try:
                        await el.press("Control+A")
                        await el.press("Delete")
                    except Exception:
                        pass
                    await el.fill(desired_title)
                    logger.info(f"[LD-Recover] ブログタイトルを設定: {desired_title} ({sel})")
                    break
        except Exception:
            logger.warning("[LD-Recover] タイトル入力欄の操作に失敗（非致命）", exc_info=True)

        # 「ブログを作成する」をクリック（blog_id は触らない）
        async def click_create():
            await page.wait_for_selector('input[type="submit"][value="ブログを作成する"]', timeout=10000)
            await page.click('input[type="submit"][value="ブログを作成する"]')
            logger.info("[LD-Recover] 『ブログを作成する』ボタンをクリック")
            # 送信直後の読み込み完了を待つ
            try:
                await page.wait_for_load_state("networkidle")
            except Exception:
                # networkidle が来なくてもURLは更新されている場合があるため続行
                pass
            logger.debug(f"[LD-Recover] after create url={page.url}")

        await click_create()

        # --- 成功判定を /welcome への遷移に変更（第一条件） ---
        success = False
        try:
            await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=15000)
            success = True
            logger.info("[LD-Recover] /welcome への遷移を確認")
        except Exception:
            # 第二条件：ウェルカム画面にある導線（文言差分も許容）
            try:
                await page.wait_for_selector('a:has-text("最初のブログを書く"), a.button:has-text("はじめての投稿")', timeout=3000)
                success = True
                logger.info("[LD-Recover] welcome 導線（ボタン）の出現を確認")
            except Exception:
                pass

        if not success:
            # 同一ページに留まっている＝何らかのバリデーションで弾かれた可能性
            html_lower = (await page.content()).lower()

            dup_or_required = any(k in html_lower for k in [
                "使用できません", "既に使われています", "重複", "invalid", "already",
                "必須", "入力してください"
            ])

            if dup_or_required:
                # 最小限のフォールバック：この時だけ blog_id を入れて再試行（最大5通り）
                base = _slugify_ascii(getattr(site, "name", None) or getattr(site, "url", None) or "blog")
                candidates = [base] + [f"{base}-{i}" for i in range(1, 6)]

                # blog_id 入力欄があるかを軽く確認（自動採番UIだと存在しない場合もある）
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
                        try:
                            if await _try_set_desired_blog_id(page, cand):
                                logger.info(f"[LD-Recover] blog_id 衝突/必須を検知 → 候補で再試行: {cand}")
                                await click_create()
                                # 再送信後の成功確認
                                try:
                                    await page.wait_for_url(_re.compile(r"/welcome($|[/?#])"), timeout=8000)
                                    success = True
                                    logger.info(f"[LD-Recover] /welcome へ遷移（blog_id={cand}）")
                                    break
                                except Exception:
                                    # まだダメなら次候補
                                    continue
                        except Exception:
                            continue

            if not success:
                # ここまで来たら失敗としてダンプ
                err_html, err_png = await _dump_error("ld_atompub_create_fail")
                logger.error("[LD-Recover] ブログ作成に失敗（タイトルのみ or 自動採番不可）")
                return {
                    "success": False,
                    "error": "blog create failed",
                    "html_path": err_html,
                    "png_path": err_png
                }

        # （任意）welcome にある導線があれば押す。無くても次の /member に進めるので best-effort
        try:
            await page.click('a:has-text("最初のブログを書く"), a.button:has-text("はじめての投稿")')
            logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")
        except Exception:
            pass

        # 6) /member に遷移して blog_id を抽出
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        blog_url = await page.get_attribute('a[title="ブログ設定"]', 'href')  # 例: /blog/king123/config/
        if not blog_url:
            err_html, err_png = await _dump_error("ld_atompub_member_fail")
            return {
                "success": False,
                "error": "member page missing blog link",
                "html_path": err_html,
                "png_path": err_png
            }
        blog_id = blog_url.split("/")[2]  # "king123"
        logger.info(f"[LD-Recover] ブログIDを取得: {blog_id}")

        # 7) 設定ページ → AtomPub 発行ページへ
        config_url = f"https://livedoor.blogcms.jp{blog_url}"
        await page.goto(config_url, wait_until="load")

        await page.wait_for_selector('a.configIdxApi[title="API Keyの発行・確認"]', timeout=10000)
        await page.click('a.configIdxApi[title="API Keyの発行・確認"]')

        await page.wait_for_load_state("load")
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

        # 8) スクショ
        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        await page.screenshot(path=success_png)
        logger.info(f"[LD-Recover] AtomPubページのスクリーンショット保存: {success_png}")

        # 9) APIキー発行
        await page.wait_for_selector('input#apiKeyIssue', timeout=10000)
        await page.click('input#apiKeyIssue')
        logger.info("[LD-Recover] 『発行する』をクリック")

        await page.wait_for_selector('button:has-text("実行")', timeout=10000)
        await page.click('button:has-text("実行")')
        logger.info("[LD-Recover] モーダルの『実行』をクリック")

        # 10) 取得（valueはJSで後から入るため、非空になるまで現在値を待つ & 1回だけ再発行リトライ）
        async def _read_endpoint_and_key():
            await page.wait_for_selector('input.input-xxlarge[readonly]', timeout=15000)
            endpoint_val = await page.locator('input.input-xxlarge[readonly]').input_value()
            await page.wait_for_selector('input#apiKey', timeout=15000)
            for _ in range(30):  # 30 * 0.5s = 15s
                key_val = (await page.locator('input#apiKey').input_value()).strip()
                if key_val:
                    return endpoint_val, key_val
                await asyncio.sleep(0.5)
            return endpoint_val, ""

        endpoint, api_key = await _read_endpoint_and_key()

        if not api_key:
            logger.warning("[LD-Recover] API Keyが空。ページを再読み込みして再発行をリトライします")
            await page.reload(wait_until="load")
            await page.wait_for_selector('input#apiKeyIssue', timeout=15000)
            await page.click('input#apiKeyIssue')
            await page.wait_for_selector('button:has-text("実行")', timeout=15000)
            await page.click('button:has-text("実行")')
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

        # 11) 返却（DB保存は呼び出し元が担当）
        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key,
            "endpoint": endpoint,
            "blog_title": desired_title  # ← 任意で追加
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
