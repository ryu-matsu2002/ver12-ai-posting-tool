import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re as _re

from app import db
from app.models import ExternalBlogAccount
from app.enums import BlogType

logger = logging.getLogger(__name__)


# ---- 文字列ユーティリティ -------------------------------------------------
def _slugify_ascii(s: str) -> str:
    """日本語/記号混じり → 半角英数とハイフンの短いスラッグ（先頭英字、最大20文字）"""
    try:
        from unidecode import unidecode  # 未導入でも動く
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


def _nice_title_from_site(site, default_slug: str) -> str:
    """カードと同じ “ai-blog” 系か、site.name をそのままタイトルに使う"""
    if getattr(site, "name", None):
        return str(site.name)[:48]
    # site.name が無いなら slug をスペース区切りに
    title = default_slug.replace("-", " ").strip()
    if not title:
        title = "My Blog"
    return title[:48]


# ---- 画面操作ユーティリティ -----------------------------------------------
async def _try_set_desired_blog_id(page, desired: str) -> bool:
    """
    ブログ作成画面で希望 blog_id を入力する。
    入力欄が見つからなくても True（サービス側が自動採番の可能性）。
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
                    logger.info(f"[LD-Recover] blog_id 欄に希望値を入力: {desired} ({sel})")
                    return True
            except Exception:
                continue
        logger.info("[LD-Recover] blog_id 入力欄が見つからず（自動採番想定）")
        return True
    except Exception:
        return False


async def _set_blog_title(page, blog_id: str, title: str) -> bool:
    """
    ブログの「タイトル」を設定（カードと同じ見栄えにする）。
    成功/見つからない場合は True、明確な失敗のみ False を返す（非致命）。
    """
    try:
        config_url = f"https://livedoor.blogcms.jp/blog/{blog_id}/config/"
        await page.goto(config_url, wait_until="load")

        # 「基本設定」タブがある場合は押す（無ければそのまま）
        try:
            # CSS が変わることがあるので候補を広く
            for sel in [
                'a.configIdxBasic[title*="基本"]',
                'a:has-text("基本設定")',
                'a[href$="/config/"]'
            ]:
                if await page.locator(sel).count() > 0:
                    await page.click(sel)
                    await page.wait_for_load_state("load")
                    break
        except Exception:
            pass

        # タイトル入力候補
        title_inputs = ['#title', '#blogTitle', 'input[name="title"]', 'input[name="blog_title"]']
        set_ok = False
        for tsel in title_inputs:
            try:
                if await page.locator(tsel).count() > 0:
                    await page.fill(tsel, title)
                    set_ok = True
                    logger.info(f"[LD-Recover] ブログタイトル入力: {title} ({tsel})")
                    break
            except Exception:
                continue

        if not set_ok:
            logger.info("[LD-Recover] タイトル入力欄が見つからず（スキップ）")
            return True

        # 保存ボタン候補
        save_buttons = [
            'input[type="submit"][value*="変更"]',
            'input[type="submit"][value*="保存"]',
            'button[type="submit"]',
            'input[name="submit"]'
        ]
        for bsel in save_buttons:
            try:
                if await page.locator(bsel).count() > 0:
                    await page.click(bsel)
                    await page.wait_for_load_state("load")
                    logger.info("[LD-Recover] タイトル保存ボタンをクリック")
                    return True
            except Exception:
                continue

        logger.info("[LD-Recover] 保存ボタンが見つからず（未保存の可能性）")
        return True
    except Exception:
        logger.warning("[LD-Recover] タイトル設定に失敗（非致命）", exc_info=True)
        return False


# ---- メイン ---------------------------------------------------------------
async def recover_atompub_key(
    page,
    nickname: str,
    email: str,
    password: str,
    site,
    desired_blog_id: str | None = None
) -> dict:
    """
    - Livedoorブログの作成 → AtomPub APIキーを発行・取得
    - desired_blog_id があればそれで作成を試み、重複時は -2, -3… と採番
    - 作成後、ブログ「タイトル」をサイト名に近い見栄えへ自動設定
    - DB保存は行わず、呼び出し元へ返す
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # 希望 blog_id の準備（無ければ site.name 等から生成）
        if not desired_blog_id:
            base_text = (getattr(site, "name", None) or getattr(site, "url", None) or "blog")
            desired_blog_id = _slugify_ascii(base_text)

        # 2) 希望 blog_id 入力（欄があれば）
        ok = await _try_set_desired_blog_id(page, desired_blog_id)
        if not ok:
            logger.warning("[LD-Recover] blog_id 入力に失敗（入力欄不在の可能性）")

        # 3) 「ブログを作成する」をクリック
        async def click_create():
            await page.wait_for_selector('input[type="submit"][value="ブログを作成する"]', timeout=10000)
            await page.click('input[type="submit"][value="ブログを作成する"]')
            logger.info("[LD-Recover] 『ブログを作成する』ボタンをクリック")

        await click_create()

        # 4) 重複時のリトライ（最大5回、-2, -3...）
        retry = 0
        while True:
            try:
                await page.wait_for_selector('a.button[href*="edit?utm_source=pcwelcome"]', timeout=5000)
                break  # 成功
            except Exception:
                content = (await page.content()).lower()
                dup_hint = any(k in content for k in ["使用できません", "既に使われています", "存在します", "重複", "invalid", "already"])
                if dup_hint and retry < 5:
                    retry += 1
                    candidate = f"{desired_blog_id}-{retry}"
                    logger.info(f"[LD-Recover] blog_id が重複の可能性 → 再試行: {candidate}")
                    await _try_set_desired_blog_id(page, candidate)
                    await click_create()
                    desired_blog_id = candidate
                    continue
                else:
                    err_html, err_png = await _dump_error("ld_atompub_create_fail")
                    logger.error("[LD-Recover] ブログ作成に失敗（重複回避不能 or 不明エラー）")
                    return {"success": False, "error": "blog create failed", "html_path": err_html, "png_path": err_png}

        # 5) 「最初のブログを書く」をクリック
        await page.click('a.button[href*="edit?utm_source=pcwelcome"]')
        logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")

        # 6) /member に遷移して blog_id を抽出
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        blog_url = await page.get_attribute('a[title="ブログ設定"]', 'href')  # 例: /blog/abc123/config/
        if not blog_url:
            err_html, err_png = await _dump_error("ld_atompub_member_fail")
            return {"success": False, "error": "member page missing blog link", "html_path": err_html, "png_path": err_png}
        blog_id = blog_url.split("/")[2]  # "abc123"
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
            return {"success": False, "error": "redirected to member", "html_path": err_html, "png_path": err_png}

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

        # 10) 取得
        await page.wait_for_selector('input.input-xxlarge[readonly]', timeout=10000)
        endpoint = await page.get_attribute('input.input-xxlarge[readonly]', 'value')
        await page.wait_for_selector('input#apiKey', timeout=10000)
        api_key = await page.get_attribute('input#apiKey', 'value')
        logger.info(f"[LD-Recover] ✅ AtomPub endpoint: {endpoint}")
        logger.info(f"[LD-Recover] ✅ AtomPub key: {api_key}")

        # 11) ブログタイトルを綺麗な名前に（非致命・ベストエフォート）
        desired_title = _nice_title_from_site(site, desired_blog_id)
        await _set_blog_title(page, blog_id, desired_title)

        # 12) 返却（DB保存は呼び出し元が担当）
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
        return {"success": False, "error": str(e), "html_path": err_html, "png_path": err_png}
