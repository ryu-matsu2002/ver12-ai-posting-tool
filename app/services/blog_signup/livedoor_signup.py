

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
from app.services.mail_utils.mail_gw import create_inbox, poll_latest_link_gw
from app.services.blog_signup.crypto_utils import encrypt
from app.services.captcha_solver import solve

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
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

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
            await browser.close()
            logger.error("[LD-Signup] CAPTCHA画像の取得に失敗しました", exc_info=True)
            raise RuntimeError("CAPTCHA画像の取得に失敗しました") from e

        await browser.close()

        return {
            "filename": filename,
            "web_path": f"/static/captchas/{filename}",
            "abs_path": str(filepath)
        }


# ──────────────────────────────────────────────
# ✅ CAPTCHA突破 + スクリーンショット付きサインアップ処理
# ──────────────────────────────────────────────
from datetime import datetime
from pathlib import Path
import asyncio
import logging
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

async def run_livedoor_signup(site, email, token, nickname, password, job_id=None):
    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import encrypt
    from app.services.mail_utils.mail_gw import poll_latest_link_gw
    from app import db
    from app.enums import BlogType

    logger.info(f"[LD-Signup] run_livedoor_signup() 実行開始: email={email}, nickname={nickname}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            logger.info("[LD-Signup] livedoor登録ページへ遷移開始")
            await page.goto("https://member.livedoor.com/register/input")
            await page.wait_for_selector('input[name="livedoor_id"]', timeout=10000)
            logger.info("[LD-Signup] フォーム表示完了、入力開始")

            await page.fill('input[name="livedoor_id"]', nickname)
            logger.info(f"[LD-Signup] 入力: livedoor_id={nickname}")

            await page.fill('input[name="password"]', password)
            await page.fill('input[name="password2"]', password)
            await page.fill('input[name="email"]', email)
            logger.info(f"[LD-Signup] 入力: email={email}")

            await page.click('input[value="ユーザー情報を登録"]')
            logger.info("[LD-Signup] [ユーザー情報を登録] ボタンクリック")

            # CAPTCHAページに遷移するのを検知し、停止
            logger.info("[LD-Signup] CAPTCHAページへの遷移を確認中...")
            for i in range(20):  # 最大60秒程度確認
                await asyncio.sleep(3)
                logger.debug(f"[LD-Signup] URLチェック中... 現在: {page.url}")
                if "captcha" in page.url or "register/captcha" in page.url:
                    logger.info("[LD-Signup] CAPTCHAページに遷移しました。ユーザーの手動入力を待機します。")
                    break

            # CAPTCHA入力完了（/register/done）まで最大10分間待機
            logger.info("[LD-Signup] CAPTCHA完了（/register/done）遷移を最大10分間待機します")
            for i in range(120):  # 10分間チェック
                await asyncio.sleep(5)
                logger.debug(f"[LD-Signup] CAPTCHA待機中... {page.url}")
                if page.url.endswith("/register/done"):
                    logger.info("[LD-Signup] CAPTCHA突破検知: /register/done に遷移済み")
                    break
            else:
                logger.warning("[LD-Signup] CAPTCHA突破未完了（/register/done に遷移せず）")
                return {"status": "captcha_not_completed"}

            # ✅ メール認証処理
            logger.info("[LD-Signup] メール認証リンク取得開始...")
            url = None
            for i in range(3):
                url = await poll_latest_link_gw(token)
                if url:
                    logger.info(f"[LD-Signup] メール認証リンク取得成功（試行{i+1}回目）: {url}")
                    break
                logger.warning(f"[LD-Signup] メール認証リンク取得失敗（試行{i+1}/3）")
                await asyncio.sleep(5)

            if not url:
                html = await page.content()
                err_html = f"/tmp/ld_email_link_fail_{timestamp}.html"
                err_png = f"/tmp/ld_email_link_fail_{timestamp}.png"
                Path(err_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=err_png)
                logger.error(f"[LD-Signup] メールリンク取得失敗 ➜ HTML: {err_html}, PNG: {err_png}")
                raise RuntimeError("確認メールリンクが取得できません（リトライ上限に到達）")

            await page.goto(url)
            await page.wait_for_timeout(2000)

            html = await page.content()
            blog_id = await page.input_value("#livedoor_blog_id")
            api_key = await page.input_value("#atompub_key")

            if not blog_id or not api_key:
                fail_html = f"/tmp/ld_final_fail_{timestamp}.html"
                fail_png = f"/tmp/ld_final_fail_{timestamp}.png"
                Path(fail_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=fail_png)
                logger.error(f"[LD-Signup] 登録後の情報取得失敗 ➜ HTML: {fail_html}, PNG: {fail_png}")
                raise RuntimeError("確認リンク遷移後に必要な値が取得できません")

            logger.info(f"[LD-Signup] 登録成功: blog_id={blog_id}, api_key=取得済み")

            success_html = f"/tmp/ld_success_{timestamp}.html"
            success_png = f"/tmp/ld_success_{timestamp}.png"
            Path(success_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=success_png)
            logger.info(f"[LD-Signup] スクリーンショット保存完了: {success_html}, {success_png}")

            # DB登録処理
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
                "status": "signup_success",
                "blog_id": blog_id,
                "api_key": api_key,
                "blog_url": f"https://blog.livedoor.jp/{blog_id}/",
                "html_path": success_html,
                "png_path": success_png
            }

        except Exception as e:
            logger.exception(f"[LD-Signup] 例外発生: {e}")
            raise

        finally:
            await browser.close()
            logger.info("[LD-Signup] ブラウザを閉じました")



# ✅ CAPTCHA送信用ステップ2関数
async def run_livedoor_signup_step2(site, email, token, nickname, password,
                                    captcha_text: str, captcha_image_path: str):
    from app.models import ExternalBlogAccount
    from app import db

    # CAPTCHA完了状態に更新
    account = db.session.query(ExternalBlogAccount).filter_by(
        site_id=site.id,
        email=email
    ).first()
    if account:
        account.is_captcha_completed = True
        db.session.commit()

    return await run_livedoor_signup(
        site=site,
        email=email,
        token=token,
        nickname=nickname,
        password=password,
        captcha_text=captcha_text,
        captcha_image_path=captcha_image_path
    )

# ✅ 新方式：GUI操作でCAPTCHAを手動入力 → /register/done を検知して再開
import asyncio
import subprocess
import json
import tempfile
from pathlib import Path

async def run_livedoor_signup_gui(site, email, token, nickname, password):
    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import encrypt
    from app.services.mail_utils.mail_gw import poll_latest_link_gw
    from app import db
    from app.enums import BlogType
    from datetime import datetime

    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 一時ファイルに入力データを保存
    temp_input = Path(tempfile.gettempdir()) / f"ld_gui_input_{timestamp}.json"
    temp_output = Path(tempfile.gettempdir()) / f"ld_gui_output_{timestamp}.json"
    temp_input.write_text(json.dumps({
        "site_id": site.id,
        "email": email,
        "token": token,
        "nickname": nickname,
        "password": password,
        "output_path": str(temp_output),
    }), encoding="utf-8")

    # xvfb-run で別スクリプトを実行
    proc = await asyncio.create_subprocess_exec(
        "/usr/local/bin/xvfb-run-wrapper", "python3", "scripts/gui_signup_runner.py", str(temp_input),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )


    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"GUI登録スクリプトが失敗しました:\n{stderr.decode()}")

    if not temp_output.exists():
        raise RuntimeError("GUI登録後の出力ファイルが存在しません")

    # 出力結果を読み取り
    result = json.loads(temp_output.read_text(encoding="utf-8"))

    blog_id = result.get("blog_id")
    api_key = result.get("api_key")
    if not blog_id or not api_key:
        raise RuntimeError("blog_id または api_key の取得に失敗しました")

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
    )
    db.session.add(account)
    db.session.commit()

    logger.info(f"[LD-GUI] 登録成功: blog_id={blog_id}")
    return {
        "status": "success",
        "blog_id": blog_id,
        "api_key": api_key,
        "blog_url": f"https://blog.livedoor.jp/{blog_id}/"
    }
