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


async def run_livedoor_signup(site, email, token, nickname, password,
                              captcha_text: str | None = None,
                              job_id=None,
                              captcha_image_path: str | None = None):
    from app.models import ExternalBlogAccount
    from app.services.blog_signup.crypto_utils import encrypt
    from app.services.mail_utils.mail_gw import poll_latest_link_gw
    from app import db
    from app.enums import BlogType

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            await page.goto("https://member.livedoor.com/register/input")
            await page.wait_for_selector('input[name="livedoor_id"]', timeout=10000)

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

            await page.wait_for_selector("#captcha-img", timeout=10000)

            # ✅ CAPTCHA画像のsrcを記録
            captcha_img_src_before = await page.get_attribute("#captcha-img", "src")
            logger.info(f"[LD-Signup] CAPTCHA画像のsrc: {captcha_img_src_before}")

            # ✅ CAPTCHA画像をimg要素から直接保存
            # ✅ CAPTCHA画像をimg要素から直接保存（トークンを使わない）
            captcha_element = await page.wait_for_selector("#captcha-img", timeout=10000)
            from uuid import uuid4
            safe_id = uuid4().hex[:8]  # ファイル名が255文字を超えないよう短縮ID
            captcha_image_path_default = f"/tmp/captcha_{safe_id}_{timestamp}.png"
            await captcha_element.screenshot(path=captcha_image_path_default)
            logger.info(f"[LD-Signup] CAPTCHA画像を保存: {captcha_image_path_default}")


            # Step 1のみを想定（画像保存して終了）
            if not captcha_text:
                logger.info("[LD-Signup] CAPTCHA解答が未指定のため、画像保存のみを行って終了します")
                return {
                    "status": "captcha_required",
                    "captcha_path": captcha_image_path_default
                }

            # Step 2の場合：保存済み画像を使って送信処理へ進む
            if not captcha_image_path or not Path(captcha_image_path).exists():
                raise RuntimeError("CAPTCHA画像ファイルが見つかりません（画像固定モード）")

            logger.info(f"[LD-Signup] CAPTCHA画像を再取得せず、固定パスを使用: {captcha_image_path}")
            captcha_text = captcha_text.replace(" ", "").replace("　", "")
            logger.info(f"[LD-Signup] CAPTCHA入力フィールドへ送信開始: {captcha_text}")
            await page.fill("#captcha", captcha_text)

            start_submit = datetime.now()
            logger.info(f"[LD-Signup] CAPTCHA入力完了、即完了ボタンをクリック")
            await page.click('input[id="commit-button"]')
            end_submit = datetime.now()
            logger.info(f"[LD-Signup] CAPTCHA入力→完了クリックまでの所要時間: {(end_submit - start_submit).total_seconds()}秒")

            await page.wait_for_timeout(2000)
            html = await page.content()
            current_url = page.url

            # CAPTCHA失敗判定
            if not ("仮登録" in html or "確認メールを送信しました" in html or current_url.endswith("/register/done")):
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

            # メール確認リンク取得
            logger.info("[LD-Signup] メール確認中...")
            url = None
            for i in range(3):
                url = await poll_latest_link_gw(token)
                if url:
                    break
                logger.warning(f"[LD-Signup] メールリンクがまだ取得できません（試行{i+1}/3）")
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
                logger.error(f"[LD-Signup] 確認リンク遷移後の失敗 ➜ HTML: {fail_html}, PNG: {fail_png}")
                raise RuntimeError("確認リンク遷移後に必要な値が取得できません")

            logger.info(f"[LD-Signup] 登録成功: blog_id={blog_id}")

            success_html = f"/tmp/ld_success_{timestamp}.html"
            success_png = f"/tmp/ld_success_{timestamp}.png"
            Path(success_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=success_png)
            logger.info(f"[LD-Signup] 成功スクリーンショット保存: {success_html}, {success_png}")

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
                "status": "captcha_success",  # ✅ 追加：UIと通信する明示的な成功ステータス
                "captcha_success": True,
                "blog_id": blog_id,
                "api_key": api_key,
                "blog_url": f"https://blog.livedoor.jp/{blog_id}/",
                "html_path": success_html,
                "png_path": success_png
            }
        except Exception as e:
            # 例外詳細ログを記録してから再送出
            err_trace_html = f"/tmp/ld_exception_{timestamp}.html"
            err_trace_png = f"/tmp/ld_exception_{timestamp}.png"
            try:
                html = await page.content()
                Path(err_trace_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=err_trace_png)
                logger.error(f"[LD-Signup] 例外時スクリーンショット保存 ➜ HTML: {err_trace_html}, PNG: {err_trace_png}")
            except Exception as inner:
                logger.warning(f"[LD-Signup] 例外処理中のスクリーンショット保存にも失敗: {inner}")

            logger.exception(f"[LD-Signup] CAPTCHA突破後の最終処理で例外発生: {e}")
            raise  # エラーを再送出

        finally:
            await browser.close()


# ✅ CAPTCHA送信用ステップ2関数
async def run_livedoor_signup_step2(site, email, token, nickname, password,
                                    captcha_text: str, captcha_image_path: str):
    return await run_livedoor_signup(
        site=site,
        email=email,
        token=token,
        nickname=nickname,
        password=password,
        captcha_text=captcha_text,
        captcha_image_path=captcha_image_path
    )
