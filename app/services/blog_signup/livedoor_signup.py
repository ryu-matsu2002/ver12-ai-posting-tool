"""
ライブドアブログ アカウント自動登録（AIエージェント仕様）
==================================
* Playwright + AIエージェントでフォーム入力 → 仮登録 → メール確認 → 本登録
* CAPTCHA対応（任意対応へ変更）, 成功判定, API Key 抽出も含む
"""

from __future__ import annotations

import asyncio
import logging
import time
import random
import string
from pathlib import Path
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.mail_utils.mail_gw import create_inbox, poll_latest_link_gw
from app.services.blog_signup.crypto_utils import encrypt
from app.services.agent.livedoor_gpt_agent import LivedoorAgent
from app.services.captcha_solver import solve  # ✅ AI CAPTCHAソルバ

logger = logging.getLogger(__name__)


def generate_safe_id(n=10) -> str:
    chars = string.ascii_lowercase + string.digits + "_"
    return ''.join(random.choices(chars, k=n))


def register_blog_account(site, email_seed: str = "ld") -> ExternalBlogAccount:
    import nest_asyncio
    nest_asyncio.apply()

    account = ExternalBlogAccount.query.filter_by(
        site_id=site.id, blog_type=BlogType.LIVEDOOR
    ).first()
    if account:
        return account

    email, token = create_inbox()
    logger.info("[LD-Signup] disposable email = %s", email)

    password = "Ld" + str(int(time.time()))
    nickname = generate_safe_id(10)

    try:
        res = asyncio.run(run_livedoor_signup(site, email, token, nickname, password))
    except Exception as e:
        logger.error("[LD-Signup] failed: %s", str(e))
        raise

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


async def run_livedoor_signup(site, email, token, nickname, password, job_id=None):
    from playwright.async_api import async_playwright, TimeoutError

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto("https://member.livedoor.com/register/input")
            assert "register/input" in page.url

            # ✅ CAPTCHA欄が存在するかチェック（最大5秒）
            try:
                await page.wait_for_selector("#captcha_text", timeout=5000)
                # CAPTCHA画像があるときだけ解く
                captcha_path = "/tmp/ld_captcha_screen.png"
                await page.screenshot(path=captcha_path)
                logger.info(f"[LD-Signup] CAPTCHA画像を保存: {captcha_path}")
                solved_text = solve(captcha_path)
                logger.info(f"[LD-Signup] solve()の推測: '{solved_text}'")
                await page.fill("#captcha_text", solved_text)
            except TimeoutError:
                logger.info("[LD-Signup] CAPTCHAが非表示のためスキップ")

            # ✅ フォーム入力処理（常に実行）
            await page.fill("#email", email)
            await page.fill("#password", password)
            await page.fill("#password-confirmation", password)
            await page.fill("#nickname", nickname)
            await page.click("#commit-button")

            await page.wait_for_timeout(2000)
            content = await page.content()

            success_patterns = [
                "メールを送信しました",
                "仮登録",
                "仮登録メールをお送りしました"
            ]
            if not any(pat in content for pat in success_patterns):
                fail_path = "/tmp/ld_signup_post_submit.html"
                Path(fail_path).write_text(content, encoding="utf-8")
                logger.error(f"[LD-Signup] CAPTCHAまたは入力失敗 → HTML保存: {fail_path}")
                raise RuntimeError("登録失敗：CAPTCHAまたはフォームエラー")

            # ✅ メール確認 → 本登録へ
            link = await poll_latest_link_gw(token=token)
            logger.info(f"[LD-Signup] メールリンク取得: {link}")

            await page.goto(link)
            await page.wait_for_timeout(1000)

            agent = LivedoorAgent(
                site=site,
                email=email,
                password=password,
                nickname=nickname,
                token=token
            )
            agent.job_id = job_id
            return await agent.run()

        except Exception as e:
            logger.exception(f"[LD-Signup] run_livedoor_signup 失敗: {e}")
            raise

        finally:
            await browser.close()
