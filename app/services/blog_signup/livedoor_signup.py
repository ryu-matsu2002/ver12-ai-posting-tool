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
    """先頭英字 + 半角英小文字 + 数字 + アンダーバー"""
    chars = string.ascii_lowercase + string.digits + "_"
    first_char = random.choice(string.ascii_lowercase)  # 先頭は英字に固定
    rest = ''.join(random.choices(chars, k=n - 1))
    return first_char + rest


def generate_safe_password(n=12) -> str:
    chars = string.ascii_letters + string.digits + "-_%$#"
    while True:
        password = ''.join(random.choices(chars, k=n))
        if any(c in "-_%$#" for c in password):  # 記号を必ず含む
            return password


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
    password = generate_safe_password()
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
# ✅ CAPTCHA画像の取得・保存だけ行う関数（ステップ①②＋学習用対応）
# ──────────────────────────────────────────────
from datetime import datetime
from pathlib import Path

# CAPTCHA画像の保存先ディレクトリ
CAPTCHA_SAVE_DIR = Path("static/captchas")
CAPTCHA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

async def prepare_livedoor_captcha(email: str, nickname: str, password: str) -> dict:
    """
    CAPTCHA画像を取得して保存し、ファイルURLとファイル名を返す（/static/captchas/...）
    → CAPTCHA突破AIの学習データにも利用可能な構成
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("https://member.livedoor.com/register/input")

        # 入力フォームを最低限埋める
        await page.fill('input[name="livedoor_id"]', nickname)
        await page.fill('input[name="password"]', password)
        await page.fill('input[name="password2"]', password)
        await page.fill('input[name="email"]', email)
        await page.click('input[value="ユーザー情報を登録"]')

        # CAPTCHA画像の表示を待機
        await page.wait_for_selector("#captcha-img", timeout=10000)
        captcha_element = await page.query_selector("#captcha-img")

        # CAPTCHA画像を保存（ファイル名を記録）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captcha_{nickname}_{timestamp}.png"
        filepath = CAPTCHA_SAVE_DIR / filename

        image_bytes = await captcha_element.screenshot()
        filepath.write_bytes(image_bytes)

        await browser.close()

        # ✅ URLとファイル名の両方を返却（後続でセッションや学習保存に利用）
        return {
            "image_url": f"/static/captchas/{filename}",
            "image_filename": filename,
        }


# ──────────────────────────────────────────────
# ✅ CAPTCHA突破 + スクリーンショット付きサインアップ処理
# ──────────────────────────────────────────────
async def run_livedoor_signup(site, email, token, nickname, password, captcha_text: str | None = None, job_id=None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            await page.goto("https://member.livedoor.com/register/input")
            try:
                await page.wait_for_selector('input[name="livedoor_id"]', timeout=10000)
            except Exception as e:
                html_path = f"/tmp/ld_wait_fail_{timestamp}.html"
                png_path = f"/tmp/ld_wait_fail_{timestamp}.png"
                html = await page.content()
                Path(html_path).write_text(html, encoding="utf-8")
                await page.screenshot(path=png_path)
                logger.error(f"[LD-Signup] ID入力欄の表示待機に失敗 ➜ HTML: {html_path}, Screenshot: {png_path}")
                raise RuntimeError("ID入力欄の読み込みに失敗（タイムアウト）") from e

            logger.info(f"[LD-Signup] 入力: id = {nickname}")
            await page.fill('input[name="livedoor_id"]', nickname)
            logger.info(f"[LD-Signup] 入力: password = {password}")
            await page.fill('input[name="password"]', password)
            logger.info(f"[LD-Signup] 入力: password2 = {password}")
            await page.fill('input[name="password2"]', password)
            logger.info(f"[LD-Signup] 入力: email = {email}")
            await page.fill('input[name="email"]', email)

            logger.info("[LD-Signup] ユーザー情報を登録ボタンをクリック")
            await page.click('input[value="ユーザー情報を登録"]')

            # ✅ CAPTCHAページ検出
            try:
                # CAPTCHA画像が表示されるまで待つ（10秒）
                await page.wait_for_selector("#captcha-img", timeout=10000)
            except Exception as e:
                html = await page.content()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = f"/tmp/ld_captcha_fail_{timestamp}.html"
                img_path = f"/tmp/ld_captcha_fail_{timestamp}.png"
                Path(html_path).write_text(html, encoding="utf-8")
                await page.screenshot(path=img_path)
                logger.error(f"[LD-Signup] CAPTCHA画像の表示に失敗 ➜ HTML: {html_path}, PNG: {img_path}")
                raise RuntimeError("CAPTCHA画像の表示に失敗（#captcha-img が見つかりません）")

            captcha_element = await page.wait_for_selector("#captcha_img")
            captcha_bytes = await captcha_element.screenshot()
            captcha_path = f"/tmp/captcha_{timestamp}.png"
            with open(captcha_path, "wb") as f:
                f.write(captcha_bytes)
            logger.info(f"[LD-Signup] CAPTCHA画像保存: {captcha_path}")

            
            if captcha_text:
                logger.info(f"[LD-Signup] CAPTCHA手入力: {captcha_text}")
            else:
                captcha_text = solve(captcha_bytes)
                logger.info(f"[LD-Signup] CAPTCHA自動推論: {captcha_text}")

            await page.fill("#captcha", captcha_text)


            logger.info("[LD-Signup] 完了ボタンをクリック")
            await page.click('input[id="commit-button"]')
            await page.wait_for_timeout(2000)

            html = await page.content()
            current_url = page.url

            # ✅ 仮登録成功判定
            if "仮登録メール" not in html and not current_url.endswith("/register/done"):
                error_html = f"/tmp/ld_signup_failed_{timestamp}.html"
                error_png  = f"/tmp/ld_signup_failed_{timestamp}.png"
                Path(error_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=error_png)
                logger.error(f"[LD-Signup] CAPTCHA失敗 ➜ HTML: {error_html}, PNG: {error_png}")
                raise RuntimeError("CAPTCHA突破失敗")

            
            logger.info("[LD-Signup] CAPTCHA突破成功")

            # ✅ メール確認リンク取得
            logger.info("[LD-Signup] メール確認中...")
            url = await poll_latest_link_gw(token)
            if not url:
                html = await page.content()
                err_html = f"/tmp/ld_email_link_fail_{timestamp}.html"
                err_png  = f"/tmp/ld_email_link_fail_{timestamp}.png"
                Path(err_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=err_png)
                logger.error(f"[LD-Signup] メールリンク取得失敗 ➜ HTML: {err_html}, PNG: {err_png}")
                raise RuntimeError("確認メールリンクが取得できません")

            await page.goto(url)
            await page.wait_for_timeout(2000)

            html = await page.content()
            if "ブログURL" not in html:
                fail_html = f"/tmp/ld_final_fail_{timestamp}.html"
                fail_png  = f"/tmp/ld_final_fail_{timestamp}.png"
                Path(fail_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=fail_png)
                logger.error(f"[LD-Signup] 確認リンク遷移後の失敗 ➜ HTML: {fail_html}, PNG: {fail_png}")
                raise RuntimeError("確認リンク遷移後に失敗")

            blog_id = await page.input_value("#livedoor_blog_id")
            api_key = await page.input_value("#atompub_key")

            logger.info(f"[LD-Signup] 登録成功: blog_id={blog_id}")

            # ✅ 成功時にもログ保存（任意）
            success_html = f"/tmp/ld_success_{timestamp}.html"
            success_png  = f"/tmp/ld_success_{timestamp}.png"
            Path(success_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=success_png)
            logger.info(f"[LD-Signup] 成功スクリーンショット保存: {success_html}, {success_png}")

            return {
                "blog_id": blog_id,
                "api_key": api_key
            }


        finally:
            await browser.close()