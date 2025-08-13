import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re as _re

from app import db
from app.models import ExternalBlogAccount
from app.enums import BlogType

logger = logging.getLogger(__name__)

# ファイル先頭の import 群の近くに追加

def _has_cjk(s: str) -> bool:
    return bool(_re.search(r"[\u3040-\u30FF\u3400-\u9FFF]", s or ""))

def _craft_blog_title(site) -> str:
    """
    サイト名をベースに、少しだけ変えたブログタイトルを作る。
    日本語名→「◯◯の記録／メモ／ラボ／ノート」等
    英語名  → 「◯◯ Journal／Notes／Lab／Digest」等
    """
    base = (getattr(site, "name", None) or getattr(site, "url", None) or "Blog").strip()
    base = base[:30]  # 長過ぎ防止

    if _has_cjk(base):
        variants = [f"{base}の記録", f"{base}メモ", f"{base}ラボ", f"{base}ノート", f"{base}ブログ"]
    else:
        variants = [f"{base} Journal", f"{base} Notes", f"{base} Lab", f"{base} Digest", f"{base} Blog"]

    # 安定的に選ぶ（毎回固定）：base の簡単なハッシュでインデックス決定
    idx = (sum(ord(c) for c in base) % len(variants))
    title = variants[idx]
    return title[:48]  # 入力欄の上限をだいたい意識

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
    - ※ 本サイトでは「ブログタイトル入力 → 作成ボタン」のみで十分なため、
        blog_id には一切触れない（サイト側の自動採番に委ねる）
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
                        await el.press("Control+A"); await el.press("Delete")
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

        await click_create()

    
        # 成功導線「最初のブログを書く」ボタンが出るかを待つ（成功のみを待つ）
        try:
            await page.wait_for_selector('a.button[href*="edit?utm_source=pcwelcome"]', timeout=15000)
        except Exception:
            err_html, err_png = await _dump_error("ld_atompub_create_fail")
            logger.error("[LD-Recover] ブログ作成に失敗（タイトルのみで遷移せず）")
            return {
                "success": False,
                "error": "blog create failed",
                "html_path": err_html,
                "png_path": err_png
            }

        # 5) 「最初のブログを書く」をクリック
        await page.click('a.button[href*="edit?utm_source=pcwelcome"]')
        logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")

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

        # 10) 取得
        await page.wait_for_selector('input.input-xxlarge[readonly]', timeout=10000)
        endpoint = await page.get_attribute('input.input-xxlarge[readonly]', 'value')

        await page.wait_for_selector('input#apiKey', timeout=10000)
        api_key = await page.get_attribute('input#apiKey', 'value')

        logger.info(f"[LD-Recover] ✅ AtomPub endpoint: {endpoint}")
        logger.info(f"[LD-Recover] ✅ AtomPub key: {api_key}")

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
