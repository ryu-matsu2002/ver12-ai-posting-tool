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
