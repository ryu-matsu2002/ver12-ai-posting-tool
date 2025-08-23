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
import json
import os
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint, render_template, redirect, url_for, flash, current_app
)

from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link_tm_async as poll_latest_link_gw
from app.services.blog_signup.crypto_utils import encrypt
from app.services.captcha_solver import solve
from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


# このモジュール用の Blueprint
bp = Blueprint("livedoor_signup", __name__, url_prefix="/livedoor-signup")

logger = logging.getLogger(__name__)

# 追加ここから
import re as _re
try:
    from unidecode import unidecode  # pip install Unidecode があれば日本語→ローマ字に
except Exception:                     # 未導入でも動くフォールバック
    def unidecode(x): return x

def _slugify_ascii(s: str) -> str:
    """日本語/記号混じり → 半角英数とハイフンの短いスラッグに（livedoor向け）"""
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

def suggest_livedoor_blog_id(base_text: str, db_session) -> str:
    """
    サイト名/ドメインの文字列から blog_id 候補を作り、
    DBに既存があれば `-2`, `-3`... と採番して一意にする
    """
    from app.models import ExternalBlogAccount
    from app.enums import BlogType

    base = _slugify_ascii(base_text)
    candidate, n = base, 0
    while True:
        exists = db_session.query(ExternalBlogAccount.id).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.livedoor_blog_id == candidate
        ).first()
        if not exists:
            return candidate
        n += 1
        tail = str(n)
        candidate = (base[: max(1, 20 - len(tail) - 1)] + "-" + tail)
# 追加ここまで


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

    # 既に登録済みなら、その内容を確認して不足があれば後で更新
    account = ExternalBlogAccount.query.filter_by(
        site_id=site.id, blog_type=BlogType.LIVEDOOR
    ).first()

    # メール生成
    email, _, token = create_inbox()
    if not email or not token:
        logger.error("[LD-Signup] JWTまたはEmailが取得できなかったため処理中断")
        raise RuntimeError("メール認証用のJWT/Email取得に失敗")

    logger.info("[LD-Signup] disposable email = %s", email)
    password = generate_safe_password()
    nickname = generate_safe_id(10)

    # サインアップ実行（DB保存はここではしない）
    res = asyncio.run(run_livedoor_signup(site, email, token, nickname, password))

    # CAPTCHA が必要だったなど、ここで続行できない場合は例外にする
    if isinstance(res, dict) and res.get("status") == "captcha_required":
        raise RuntimeError("CAPTCHAが必要なため自動登録を中断しました（手動対応 or 別フローへ）")

    blog_id = res.get("blog_id")
    api_key = res.get("api_key")
    if not blog_id or not api_key:
        raise RuntimeError("livedoor サインアップ結果に blog_id / api_key が含まれていません")

    # ここでのみ DB 保存（アップサート）
    endpoint = os.getenv("LIVEDOOR_ATOM_ENDPOINT", "https://livedoor.blogcms.jp/atompub")

    if account is None:
        account = ExternalBlogAccount(
            site_id=site.id,
            blog_type=BlogType.LIVEDOOR,
            email=email,
            username=blog_id,              # ← 投稿側の想定に合わせる
            password=password,
            nickname=nickname,
            livedoor_blog_id=blog_id,
            atompub_key_enc=encrypt(api_key),  # ← 暗号化版（安全に参照する実装がある場合に備える）
            atompub_endpoint=endpoint,   # ← ここを endpoint ではなく atompub_endpoint
            api_post_enabled=True,
        )
        db.session.add(account)
    else:
        # 既存行を更新（不足項目を必ず埋める）
        account.email = account.email or email
        account.username = blog_id
        account.password = account.password or password
        account.nickname = account.nickname or nickname
        account.livedoor_blog_id = blog_id
        account.atompub_key_enc = encrypt(api_key)
        account.atompub_endpoint = endpoint  # ← 同上
        account.api_post_enabled = True

    db.session.commit()
    logger.info(f"[LD-Signup] アカウントをDBに保存しました（id={account.id}, blog_id={blog_id}）")
    return account



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
            headless=True,
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


            return {
                "blog_id": blog_id,
                "api_key": api_key,
                "captcha_success": bool(captcha_text)
            }

        finally:
            await context.close()
            await browser.close()

from app.services.playwright_controller import store_session

async def launch_livedoor_and_capture_captcha(
    email: str, nickname: str, password: str, session_id: str,
    desired_blog_id: str | None = None  # ★ 追加
) -> dict:

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
                                      password: str, token: str, site,
                                      desired_blog_id: str | None = None) -> dict:
    """
    CAPTCHAを入力・送信し、メール認証・アカウント作成を完了。
    - #captcha を「見える・編集可」まで待機
    - 送信ボタンは複数セレクタでフォールバック
    - 押下後は成功/失敗DOM or networkidle で確定判定
    """
    from datetime import datetime
    from pathlib import Path
    import asyncio
    import re

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 1) CAPTCHAの入力欄を確実に待つ
        logger.info("[LD-Signup] CAPTCHA入力待機開始")
        await page.wait_for_selector("#captcha", state="visible", timeout=15000)
        c_input = page.locator("#captcha")
        await c_input.wait_for(state="visible", timeout=5000)
        await c_input.focus()
        await page.wait_for_timeout(150)  # 微小ディレイ

        # 2) 入力（全選択→上書き）
        norm_text = (captcha_text or "").replace(" ", "").replace("　", "")
        logger.info(f"[LD-Signup] CAPTCHA手入力（整形後）: {norm_text}")
        await c_input.fill("")                 # 空に
        # OS差異回避のため二段構え
        try:
            await page.keyboard.press("Control+A")
        except Exception:
            try:
                await page.keyboard.press("Meta+A")
            except Exception:
                pass
        await c_input.type(norm_text, delay=50)

        # 3) 送信ボタン探索（複数候補）
        submit_candidates = [
            '#commit-button',
            'input#commit-button',
            'input[name="commit"]',
            'input[type="submit"][value="登録"]',
            'input[type="submit"]',
        ]
        commit = None
        for sel in submit_candidates:
            loc = page.locator(sel)
            if await loc.count():
                try:
                    await loc.first.wait_for(state="visible", timeout=3000)
                    commit = loc.first
                    break
                except Exception:
                    pass

        if commit:
            logger.info("[LD-Signup] 完了ボタンをクリック")
            await commit.click()
        else:
            # フォーム submit フォールバック
            logger.warning("[LD-Signup] 送信ボタン不検出 → form.submit() フォールバック")
            await page.evaluate("""
                () => {
                  const f = document.querySelector('form');
                  if (f) f.submit();
                }
            """)

        # 4) 押下後の状態確定（成功/失敗/不明）
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        html = await page.content()
        url_now = page.url

        success_re = re.compile(r"(仮登録メール|確認メール)", re.S)
        error_candidates = [
            ".error", ".errors", ".alert", ".attention", "#error", "#errors",
            "div:has-text('CAPTCHA')",
        ]

        # まずは即時判定
        if (success_re.search(html) is None) and (not url_now.endswith("/register/done")):
            # 失敗DOMの有無
            for sel in error_candidates:
                try:
                    if await page.locator(sel).is_visible():
                        fail_html = f"/tmp/ld_captcha_failed_{timestamp}.html"
                        fail_png  = f"/tmp/ld_captcha_failed_{timestamp}.png"
                        Path(fail_html).write_text(html, encoding="utf-8")
                        await page.screenshot(path=fail_png, full_page=True)
                        logger.warning(f"[LD-Signup] CAPTCHA失敗DOM検出 ➜ HTML: {fail_html}, PNG: {fail_png}")
                        return {"captcha_success": False, "error": "CAPTCHA突破失敗",
                                "html_path": fail_html, "png_path": fail_png}
                except Exception:
                    pass

            # もう少しだけ待って最終評価
            await page.wait_for_timeout(1500)
            html = await page.content()
            url_now = page.url
            if (success_re.search(html) is None) and (not url_now.endswith("/register/done")):
                unk_html = f"/tmp/ld_captcha_post_unknown_{timestamp}.html"
                unk_png  = f"/tmp/ld_captcha_post_unknown_{timestamp}.png"
                Path(unk_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=unk_png, full_page=True)
                logger.error(f"[LD-Signup] CAPTCHA送信後の状態不明 ➜ HTML: {unk_html}, PNG: {unk_png}")
                return {"captcha_success": False, "error": "CAPTCHA送信後の状態不明",
                        "html_path": unk_html, "png_path": unk_png}

        logger.info("[LD-Signup] CAPTCHA突破 → メール確認へ")

        # 5) メール確認リンク取得（少し余裕を持って待つ）
        url = None
        for i in range(4):
            u = await poll_latest_link_gw(token)
            if u:
                url = u
                break
            await asyncio.sleep(5)

        if not url:
            html = await page.content()
            err_html = f"/tmp/ld_email_link_fail_{timestamp}.html"
            err_png  = f"/tmp/ld_email_link_fail_{timestamp}.png"
            Path(err_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=err_png, full_page=True)
            logger.error(f"[LD-Signup] メールリンク取得失敗 ➜ HTML: {err_html}, PNG: {err_png}")
            return {"captcha_success": False, "error": "メール認証に失敗"}

        await page.goto(url)
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        logger.info(f"[LD-Signup] ✅ 認証リンクにアクセス成功: {url}")

        # 6) AtomPub recover（既存処理）
        recover_result = await recover_atompub_key(
            page, nickname, email, password, site, desired_blog_id=desired_blog_id
        )
        if not recover_result["success"]:
            return {
                "captcha_success": True,
                "error": recover_result.get("error", "AtomPub再取得に失敗"),
                "html_path": recover_result.get("html_path"),
                "png_path": recover_result.get("png_path"),
            }

        return {
            "captcha_success": True,
            "blog_id": recover_result["blog_id"],
            "api_key": recover_result["api_key"],
        }

    except Exception as e:
        logger.exception("[LD-Signup] CAPTCHA送信 or 登録処理で例外")
        # 例外時もダンプ
        try:
            html = await page.content()
            ex_html = f"/tmp/ld_captcha_exception_{timestamp}.html"
            ex_png  = f"/tmp/ld_captcha_exception_{timestamp}.png"
            Path(ex_html).write_text(html, encoding="utf-8")
            await page.screenshot(path=ex_png, full_page=True)
            return {"captcha_success": False, "error": str(e),
                    "html_path": ex_html, "png_path": ex_png}
        except Exception:
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



@bp.route('/confirm_email_manual/<task_id>')
def confirm_email_manual(task_id):
    """
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面。
    """
    # メール受信（最大30回ポーリング） ← 既存関数を再利用
    email_body = poll_latest_link_gw(task_id=task_id, max_attempts=30, interval=5)

    if email_body:
        # 認証URLを抽出
        verification_url = extract_verification_url(email_body)
        if verification_url:
            return render_template("confirm_email.html", verification_url=verification_url)
        else:
            flash("認証リンクが見つかりませんでした", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("認証メールを取得できませんでした", "danger")
        return redirect(url_for('dashboard'))