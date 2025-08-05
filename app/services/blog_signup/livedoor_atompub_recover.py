import asyncio
from datetime import datetime
from pathlib import Path
import logging

from app.models import ExternalBlogAccount
from app.services.blog_signup.crypto_utils import encrypt
from app import db
from app.enums import BlogType

logger = logging.getLogger(__name__)

async def recover_atompub_key(page, nickname: str, email: str, password: str, site) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # ブログ作成ページへ
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")

        await page.click('input[type="submit"][value="ブログを作成する"]')
        logger.info("[LD-Recover] 『ブログを作成する』ボタンをクリック完了")

        await page.wait_for_selector('a.button[href*="edit?utm_source=pcwelcome"]', timeout=10000)
        await page.click('a.button[href*="edit?utm_source=pcwelcome"]')
        logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")

        # ✅ /member に遷移してブログIDを抽出
        await page.goto("https://livedoor.blogcms.jp/member/", wait_until="load")
        blog_url = await page.get_attribute('a[title="ブログ設定"]', 'href')  # 例: /blog/king1234567890/config/
        blog_id = blog_url.split("/")[2]  # "king1234567890"
        logger.info(f"[LD-Recover] ブログIDを取得: {blog_id}")

        # configページへ移動して AtomPub ボタンをクリック
        config_url = f"https://livedoor.blogcms.jp{blog_url}"
        await page.goto(config_url, wait_until="load")

        await page.wait_for_selector('a.configIdxApi[title="API Keyの発行・確認"]', timeout=10000)
        await page.click('a.configIdxApi[title="API Keyの発行・確認"]')

        # AtomPubページに遷移 → URLチェック
        await page.wait_for_load_state("load")
        logger.info(f"[LD-Recover] AtomPub設定ページに遷移: {page.url}")

        if "member" in page.url:
            logger.error(f"[LD-Recover] AtomPubページが開けずに /member にリダイレクト: {page.url}")
            html = await page.content()
            error_html = f"/tmp/ld_atompub_fail_{timestamp}.html"
            error_png = f"/tmp/ld_atompub_fail_{timestamp}.png"
            Path(error_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=error_png)
            return {
                "success": False,
                "error": "Redirected to member page instead of atompub",
                "html_path": error_html,
                "png_path": error_png
            }
        
        # ✅ スクリーンショットを保存（正常ページであることが確認されたあと）
        success_png = f"/tmp/ld_atompub_page_{timestamp}.png"
        await page.screenshot(path=success_png)
        logger.info(f"[LD-Recover] AtomPubページのスクリーンショットを保存: {success_png}")

        # 発行ボタン（input#apiKeyIssue）をクリック
        await page.wait_for_selector('input#apiKeyIssue', timeout=10000)
        await page.click('input#apiKeyIssue')
        logger.info("[LD-Recover] 『発行する』ボタンをクリック")

        # ポップアップの「実行」ボタンをクリック
        await page.wait_for_selector('button:has-text("実行")', timeout=10000)
        await page.click('button:has-text("実行")')
        logger.info("[LD-Recover] モーダルの『実行』ボタンをクリック")

        # APIキー取得（inputのvalue属性を取得）
        await page.wait_for_selector('input.input-xxlarge[readonly]', timeout=10000)
        endpoint = await page.get_attribute('input.input-xxlarge[readonly]', 'value')

        await page.wait_for_selector('input#apiKey', timeout=10000)
        api_key = await page.get_attribute('input#apiKey', 'value')

        logger.info(f"[LD-Recover] ✅ AtomPubエンドポイント: {endpoint}")
        logger.info(f"[LD-Recover] ✅ AtomPubパスワード取得成功: {api_key}")



        # DB保存
        account = ExternalBlogAccount(
            site_id=site.id,
            blog_type=BlogType.LIVEDOOR,
            email=email,
            username=blog_id,
            password=password,
            nickname=nickname,
            livedoor_blog_id=blog_id,
            atompub_key_enc=encrypt(api_key),
            atompub_endpoint=endpoint,  # ← NEW
            api_post_enabled=True  # ← 追加済み
        )
        db.session.add(account)
        db.session.commit()
        logger.info(f"[LD-Recover] ✅ DB登録完了 blog_id={blog_id}")

        return {
            "success": True,
            "blog_id": blog_id,
            "api_key": api_key
        }

    except Exception as e:
        html = await page.content()
        error_html = f"/tmp/ld_atompub_fail_{timestamp}.html"
        error_png = f"/tmp/ld_atompub_fail_{timestamp}.png"
        Path(error_html).write_text(html, encoding="utf-8")
        await page.screenshot(path=error_png)
        logger.error(f"[LD-Recover] AtomPubページでエラー ➜ {error_html}, {error_png}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "html_path": error_html,
            "png_path": error_png
        }
