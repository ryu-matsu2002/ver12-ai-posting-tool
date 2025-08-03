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

        # 「発行する」ボタンをクリック（新セレクタ）
        await page.click('button:has-text("発行する")')
        logger.info("[LD-Recover] 『発行する』ボタンをクリック")

        # モーダル「実行」ボタン
        await page.wait_for_selector('button.btn-confirm', timeout=10000)
        await page.click('button.btn-confirm')
        logger.info("[LD-Recover] モーダルの『実行』ボタンをクリック")

        # APIキーの表示要素が出るのを待つ
        await page.wait_for_selector('pre.apikey', timeout=10000)
        api_key = await page.inner_text('pre.apikey')
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
