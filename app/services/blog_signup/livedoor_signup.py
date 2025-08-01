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
from flask import url_for
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link_tm_async as poll_latest_link_gw
from app.services.blog_signup.crypto_utils import encrypt
from app.services.captcha_solver import solve
from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
from playwright.async_api import async_playwright
from flask import current_app


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
    email, _, token = create_inbox()
    if not email or not token:
        logger.error("[LD-Signup] JWTまたはEmailが取得できなかったため処理中断")
        return {"captcha_success": False, "error": "メール認証用のJWT取得に失敗"}
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
# ✅ CAPTCHA画像の取得・保存（base64形式 or 画像URL両対応）
# ──────────────────────────────────────────────
async def prepare_livedoor_captcha(email: str, nickname: str, password: str) -> dict:
    """
    CAPTCHA画像を取得して保存し、ファイル名・Webパス・絶対パスを返す
    """
    from playwright.async_api import async_playwright
    from flask import current_app
    from datetime import datetime
    import asyncio
    from pathlib import Path

    CAPTCHA_SAVE_DIR = Path(current_app.root_path) / "static" / "captchas"
    CAPTCHA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # ⬅️ 人間操作に見せかける
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['ja-JP', 'ja'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            """
        )

        await page.goto("https://member.livedoor.com/register/input")

        await page.fill('input[name="livedoor_id"]', nickname)
        await page.fill('input[name="password"]', password)
        await page.fill('input[name="password2"]', password)
        await page.fill('input[name="email"]', email)

        await page.click('input[value="ユーザー情報を登録"]')
        await page.wait_for_selector("#captcha-img", state="visible", timeout=10000)
        await asyncio.sleep(0.5)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captcha_{nickname}_{timestamp}.png"
        filepath = CAPTCHA_SAVE_DIR / filename

        try:
            captcha_element = page.locator("#captcha-img")
            await captcha_element.screenshot(path=str(filepath))
            logger.info(f"[LD-Signup] CAPTCHA画像保存完了: {filepath}")
        except Exception as e:
            await context.close()
            logger.error("[LD-Signup] CAPTCHA画像の取得に失敗しました", exc_info=True)
            raise RuntimeError("CAPTCHA画像の取得に失敗しました") from e

        await context.close()

        return {
            "filename": filename,
            "web_path": f"/static/captchas/{filename}",
            "abs_path": str(filepath)
        }

# ──────────────────────────────────────────────
# ✅ CAPTCHA突破 + スクリーンショット付きサインアップ処理
# ──────────────────────────────────────────────
async def run_livedoor_signup(site, email, token, nickname, password,
                              captcha_text: str | None = None,
                              job_id=None,
                              captcha_image_path: str | None = None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=150,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="ja-JP"
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['ja-JP', 'ja'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        page = await context.new_page()


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            await page.goto("https://member.livedoor.com/register/input", wait_until="load")
            await page.wait_for_selector('input[name="livedoor_id"]', timeout=10000)

            logger.info(f"[LD-Signup] 入力: id = {nickname}")
            await page.type('input[name="livedoor_id"]', nickname, delay=100)

            logger.info(f"[LD-Signup] 入力: password = {password}")
            await page.type('input[name="password"]', password, delay=100)

            logger.info(f"[LD-Signup] 入力: password2 = {password}")
            await page.type('input[name="password2"]', password, delay=100)

            logger.info(f"[LD-Signup] 入力: email = {email}")
            await page.type('input[name="email"]', email, delay=100)

            await page.wait_for_timeout(1000)

            logger.info("[LD-Signup] ユーザー情報を登録ボタンをクリック")
            await page.click('input[value="ユーザー情報を登録"]', delay=100)

            # CAPTCHA検出
            try:
                await page.wait_for_selector("#captcha-img", timeout=5000)
                logger.info(f"[LD-Signup] CAPTCHA出現確認 → URL返却: {page.url}")
                return {
                    "status": "captcha_required",
                    "captcha_url": page.url,
                    "email": email,
                    "nickname": nickname,
                    "password": password
                }
            except Exception:
                logger.info("[LD-Signup] CAPTCHAは表示されませんでした（通常通過）")

            # ✅ CAPTCHA画像存在チェック（submit時）
            if captcha_image_path:
                if not Path(captcha_image_path).exists():
                    raise RuntimeError("CAPTCHA画像ファイルが見つかりません（画像固定モード）")
                logger.info(f"[LD-Signup] CAPTCHA画像を再取得せず、固定パスを使用: {captcha_image_path}")

            # ✅ CAPTCHAテキストがある場合は処理継続
            if captcha_text:
                captcha_text = captcha_text.replace(" ", "").replace("　", "")
                logger.info(f"[LD-Signup] CAPTCHA手入力（整形後）: {captcha_text}")
                await page.fill("#captcha", captcha_text)

                logger.info("[LD-Signup] 完了ボタンをクリック")
                await page.click('input[id="commit-button"]')
                await page.wait_for_timeout(2000)

                html = await page.content()
                current_url = page.url

                if "仮登録メール" not in html and not current_url.endswith("/register/done"):
                    error_html = f"/tmp/ld_signup_failed_{timestamp}.html"
                    error_png = f"/tmp/ld_signup_failed_{timestamp}.png"
                    Path(error_html).write_text(html, encoding="utf-8")
                    await page.screenshot(path=error_png)
                    logger.error(f"[LD-Signup] CAPTCHA失敗 ➜ HTML: {error_html}, PNG: {error_png}")

                    return {
                        "status": "captcha_failed",
                        "error": "CAPTCHA突破失敗",
                        "html_path": error_html,
                        "png_path": error_png
                    }

                logger.info("[LD-Signup] CAPTCHA突破成功")

            
            # ✅ メール確認リンク取得
            logger.info("[LD-Signup] メール確認中...")
            from app.services.mail_utils.mail_tm import poll_latest_link_tm_async

            await asyncio.sleep(10)  # livedoorからのメール送信を待つ（確実性向上）
            url = await poll_latest_link_tm_async(token)

            if not url:
                html = await page.content()
                err_html = f"/tmp/ld_email_link_fail_{timestamp}.html"
                err_png = f"/tmp/ld_email_link_fail_{timestamp}.png"
                Path(err_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=err_png)
                logger.error(f"[LD-Signup] メールリンク取得失敗 ➜ HTML: {err_html}, PNG: {err_png}")
                raise RuntimeError("確認メールリンクが取得できません（poll_latest_link_gw = None）")

            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            logger.info(f"[LD-Signup] ✅ 認証リンクにアクセス成功: {url}")



            html = await page.content()
            blog_id = await page.input_value("#livedoor_blog_id")
            api_key = await page.input_value("#atompub_key")

            if not blog_id or not api_key:
                fail_html = f"/tmp/ld_final_fail_{timestamp}.html"
                fail_png = f"/tmp/ld_final_fail_{timestamp}.png"
                Path(fail_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=fail_png)
                logger.error(f"[LD-Signup] 確認リンク遷移後の失敗 ➜ HTML: {fail_html}, PNG: {fail_png}")
                raise RuntimeError("確認リンク遷移後に必要な値が取得できません")

            logger.info(f"[LD-Signup] 登録成功: blog_id={blog_id}")

            success_html = f"/tmp/ld_success_{timestamp}.html"
            success_png = f"/tmp/ld_success_{timestamp}.png"
            Path(success_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=success_png)
            logger.info(f"[LD-Signup] 成功スクリーンショット保存: {success_html}, {success_png}")

            # DB保存処理
            from app.models import ExternalBlogAccount
            from app.services.blog_signup.crypto_utils import encrypt
            from app import db
            from app.enums import BlogType

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
            logger.info(f"[LD-Signup] アカウントをDBに保存しました（id={account.id}）")

            return {
                "blog_id": blog_id,
                "api_key": api_key,
                "captcha_success": bool(captcha_text)
            }

        finally:
            await context.close()
            await browser.close()

from app.services.playwright_controller import store_session

async def launch_livedoor_and_capture_captcha(email: str, nickname: str, password: str, session_id: str) -> dict:
    """
    CAPTCHA画像を取得し、Playwrightセッションを保持して返す。
    """
    from playwright.async_api import async_playwright
    from pathlib import Path
    from datetime import datetime

    CAPTCHA_SAVE_DIR = Path(current_app.root_path) / "static" / "captchas"
    CAPTCHA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()

    try:
        await page.goto("https://member.livedoor.com/register/input")
        await page.fill('input[name="livedoor_id"]', nickname)
        await page.fill('input[name="password"]', password)
        await page.fill('input[name="password2"]', password)
        await page.fill('input[name="email"]', email)
        await page.click('input[value="ユーザー情報を登録"]')

        await page.wait_for_selector("#captcha-img", timeout=10000)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captcha_{session_id}_{timestamp}.png"
        filepath = CAPTCHA_SAVE_DIR / filename

        await page.locator("#captcha-img").screenshot(path=str(filepath))
        logger.info(f"[LD-Signup] CAPTCHA画像を {filepath} に保存")

        # ✅ セッションに page を保持（ブラウザは閉じずに返す）
        await store_session(session_id, page)

        return {"filename": filename}

    except Exception as e:
        await browser.close()
        logger.exception("[LD-Signup] CAPTCHA画像取得失敗")
        raise RuntimeError("CAPTCHA画像の取得に失敗しました")

async def submit_captcha_and_complete(page, captcha_text: str, email: str, nickname: str,
                                      password: str, token: str, site) -> dict:
    """
    CAPTCHAを入力・送信し、メール認証・アカウント作成を完了。
    """
    from datetime import datetime
    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import encrypt
    from app import db
    from app.enums import BlogType

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # ✅ CAPTCHA送信
        logger.info(f"[LD-Signup] CAPTCHA入力中: {captcha_text}")
        await page.fill("#captcha", captcha_text)
        await page.click('input[id="commit-button"]')
        await page.wait_for_timeout(2000)

        html = await page.content()
        current_url = page.url

        # CAPTCHA突破判定
        if "仮登録メール" not in html and not current_url.endswith("/register/done"):
            error_html = f"/tmp/ld_captcha_failed_{timestamp}.html"
            error_png = f"/tmp/ld_captcha_failed_{timestamp}.png"
            Path(error_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=error_png)
            logger.warning(f"[LD-Signup] CAPTCHA失敗 ➜ HTML: {error_html}, PNG: {error_png}")
            return {
                "captcha_success": False,
                "html_path": error_html,
                "png_path": error_png
            }

        # ✅ メール認証リンク取得
        logger.info("[LD-Signup] メール確認リンク取得中...")
        url = None
        for i in range(3):
            logger.info(f"[LD-Signup] メール取得リトライ {i+1}/3")
            u = await poll_latest_link_gw(token)
            if u:
                url = u
                break
            await asyncio.sleep(5)

        if not url:
            html = await page.content()
            err_html = f"/tmp/ld_email_link_fail_{timestamp}.html"
            err_png = f"/tmp/ld_email_link_fail_{timestamp}.png"
            Path(err_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=err_png)
            logger.error(f"[LD-Signup] メールリンク取得失敗 ➜ HTML: {err_html}, PNG: {err_png}")
            return {"captcha_success": False, "error": "メール認証に失敗"}

        # ✅ 認証リンクへ遷移
        await page.goto(url)
        await page.wait_for_timeout(2000)

        # ✅ AtomPub情報取得（直接recover_atompub_keyで処理）
        logger.info("[LD-Signup] AtomPub作成ルートに移行")
        recover_result = await recover_atompub_key(page, nickname, email, password, site)

        if not recover_result["success"]:
            return {
                "captcha_success": True,
                "error": recover_result.get("error", "AtomPub再取得に失敗"),
                "html_path": recover_result.get("html_path"),
                "png_path": recover_result.get("png_path")
            }

        # ✅ DB登録
        account = ExternalBlogAccount(
            site_id=site.id,
            blog_type=BlogType.LIVEDOOR,
            email=email,
            username=recover_result["blog_id"],
            password=password,
            nickname=nickname,
            livedoor_blog_id=recover_result["blog_id"],
            atompub_key_enc=encrypt(recover_result["api_key"]),
        )
        db.session.add(account)
        db.session.commit()
        logger.info(f"[LD-Signup] アカウントDB登録完了 blog_id={recover_result['blog_id']}")

        return {
            "captcha_success": True,
            "blog_id": recover_result["blog_id"],
            "api_key": recover_result["api_key"]
        }

    except Exception as e:
        logger.exception("[LD-Signup] CAPTCHA送信 or 登録失敗")
        return {"captcha_success": False, "error": str(e)}


import re

def extract_verification_url(email_body: str) -> str | None:
    """
    livedoorの認証URLをメール本文から抽出する。
    """
    # livedoor登録メールに含まれる認証リンクのパターン
    pattern = r"https://member\.livedoor\.com/verify/[a-zA-Z0-9]+"
    match = re.search(pattern, email_body)
    if match:
        return match.group(0)
    return None

import json
import os

TEMP_DIR = "/tmp/livedoor_tasks"  # 必要なら /var/www/... 配下に移動可

os.makedirs(TEMP_DIR, exist_ok=True)

def save_livedoor_credentials(task_id: str, blog_id: str, api_key: str):
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    with open(path, "w") as f:
        json.dump({"blog_id": blog_id, "api_key": api_key}, f)

def fetch_livedoor_credentials(task_id: str) -> dict | None:
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)