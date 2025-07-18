"""
ライブドアブログ アカウント自動登録（AIエージェント仕様）
==================================
* Playwright + AIエージェントでフォーム入力 → 仮登録 → メール確認 → 本登録
* CAPTCHA対応, 成功判定, API Key 抽出も含む
"""

from __future__ import annotations

import asyncio
import logging
import time
import random
import string
from datetime import datetime
from pathlib import Path

from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.mail_utils.mail_gw import create_inbox, poll_latest_link_gw
from app.services.blog_signup.crypto_utils import encrypt
from app.services.captcha_solver import solve

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

def generate_safe_id(n=10) -> str:
    """半角英小文字 + 数字 + アンダーバー のみで構成されたID"""
    chars = string.ascii_lowercase + string.digits + "_"
    return ''.join(random.choices(chars, k=n))


def register_blog_account(site, email_seed: str = "ld") -> ExternalBlogAccount:
    import nest_asyncio
    nest_asyncio.apply()

    # 既に登録済みなら再利用
    account = ExternalBlogAccount.query.filter_by(
        site_id=site.id, blog_type=BlogType.LIVEDOOR
    ).first()
    if account:
        return account

    # メール生成
    email, token = create_inbox()
    logger.info("[LD-Signup] disposable email = %s", email)

    # パスワードは一意に
    password = "Ld" + str(int(time.time()))
    nickname = generate_safe_id(10)

    try:
        res = asyncio.run(run_livedoor_signup(site, email, token, nickname, password))
    except Exception as e:
        logger.error("[LD-Signup] failed: %s", str(e))
        raise

    # DB登録
    new_account = ExternalBlogAccount(
        site_id=site.id,
        blog_type=BlogType.LIVEDOOR,
        email=email,
        username=nickname,
        password=password,
        livedoor_blog_id=res["blog_id"],
        atompub_key_enc=encrypt(res["api_key"]),
        api_post_enabled=True,
        nickname=nickname,
    )
    db.session.add(new_account)
    db.session.commit()
    return new_account


def signup(site, email_seed: str = "ld"):
    return register_blog_account(site, email_seed=email_seed)


# ──────────────────────────────────────────────
# ✅ CAPTCHA突破 + スクリーンショット付きサインアップ処理
# ──────────────────────────────────────────────
async def run_livedoor_signup(site, email, token, nickname, password, job_id=None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto("https://member.livedoor.com/register/input")

            # フォーム入力
            await page.wait_for_selector('input[name="nickname"]', timeout=10000)
            await page.fill("#register_id", nickname)
            await page.wait_for_selector('input[name="nickname"]', timeout=10000)
            await page.fill("#register_pw", password)
            await page.wait_for_selector('input[name="nickname"]', timeout=10000)
            await page.fill("#register_pw2", password)
            await page.wait_for_selector('input[name="nickname"]', timeout=10000)
            await page.fill("#register_mail", email)

            # CAPTCHA取得と推論
            captcha_element = await page.wait_for_selector("#captcha_img")
            captcha_bytes = await captcha_element.screenshot()

            # ✅ 保存＆ログ出力
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            captcha_path = f"/tmp/captcha_{timestamp}.png"
            with open(captcha_path, "wb") as f:
                f.write(captcha_bytes)
            logger.info(f"[LD-Signup] CAPTCHA画像保存: {captcha_path}")

            captcha_text = solve(captcha_bytes)
            logger.info(f"[LD-Signup] CAPTCHA推論結果: {captcha_text}")

            await page.fill("#captcha", captcha_text)
            await page.click("#commit-button")
            await page.wait_for_timeout(2000)

            html = await page.content()
            current_url = page.url

            # ✅ 成功判定（仮登録完了 or register/done）
            if "仮登録メール" not in html and not current_url.endswith("/register/done"):
                error_html = f"/tmp/ld_signup_failed_{timestamp}.html"
                error_png  = f"/tmp/ld_signup_failed_{timestamp}.png"
                Path(error_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=error_png)
                logger.error(f"[LD-Signup] CAPTCHA失敗 ➜ HTML: {error_html}, PNG: {error_png}")
                raise RuntimeError("CAPTCHA突破失敗")

            logger.info("[LD-Signup] CAPTCHA突破成功")

            # ✅ メールから確認リンクを取得
            logger.info("[LD-Signup] メール確認中...")
            url = await poll_latest_link_gw(token)
            if not url:
                raise RuntimeError("確認メールリンクが取得できません")

            await page.goto(url)
            await page.wait_for_timeout(2000)

            html = await page.content()
            if "ブログURL" not in html:
                raise RuntimeError("確認リンク遷移後に失敗")

            # ✅ APIキーの抽出
            blog_id = await page.input_value("#livedoor_blog_id")
            api_key = await page.input_value("#atompub_key")

            return {
                "blog_id": blog_id,
                "api_key": api_key
            }

        finally:
            await browser.close()
