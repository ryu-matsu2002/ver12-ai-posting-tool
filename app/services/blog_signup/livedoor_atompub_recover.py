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
    blog_id = nickname + "g"

    try:
        logger.info("[LD-Recover] ブログ作成ページに遷移")
        await page.goto("https://livedoor.blogcms.jp/member/blog/create", wait_until="load")

        await page.click('input[type="submit"][value="ブログを作成する"]')
        logger.info("[LD-Recover] 『ブログを作成する』ボタンをクリック完了")

        await page.wait_for_selector('a.button[href*="edit?utm_source=pcwelcome"]', timeout=10000)
        await page.click('a.button[href*="edit?utm_source=pcwelcome"]')
        logger.info("[LD-Recover] 『最初のブログを書く』ボタンをクリック完了")

        atompub_url = f"https://livedoor.blogcms.jp/blog/{blog_id}/config/atompub/"
        await page.goto(atompub_url, wait_until="load")
        logger.info(f"[LD-Recover] AtomPub設定ページに遷移: {atompub_url}")

        # ⬇️ ステップ1: 「発行する」ボタン
        await page.wait_for_selector('input#apiKeyIssue', timeout=10000)
        await page.click('input#apiKeyIssue')
        logger.info("[LD-Recover] AtomPub画面の『発行する』ボタンをクリック")

        # ⬇️ ステップ2: モーダルの「発行」ボタン
        await page.wait_for_selector('input[type="button"][value="発行"]', timeout=10000)
        await page.click('input[type="button"][value="実行"]')
        logger.info("[LD-Recover] モーダルの『実行』ボタンをクリック")

        await page.wait_for_selector('input[type="text"]', timeout=10000)
        api_key = await page.input_value('input[type="text"]')
        logger.info(f"[LD-Recover] ✅ AtomPubパスワード取得成功: {api_key}")

        # ✅ DB保存（上書き or 新規）
        account = ExternalBlogAccount(
            site_id=site.id,
            blog_type=BlogType.LIVEDOOR,
            email=email,
            username=blog_id,
            password=password,
            nickname=nickname,
            livedoor_blog_id=blog_id,
            atompub_key_enc=encrypt(api_key),
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
