"""
ライブドアブログ アカウント自動登録（AIエージェント仕様）
==================================
* Playwright を長寿命コントローラ（pwctl）で管理
* 2段階フロー:
  - prepare_captcha(): 入力→CAPTCHA画像の保存（セッション保持）
  - submit_captcha(): CAPTCHA送信→/register/done待機→（以降はメール確認/キー回収の差込点）
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from flask import Blueprint, render_template, redirect, url_for, flash
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount

from app.services.pw_controller import pwctl  # ← 長寿命Playwright
from playwright.async_api import Page, TimeoutError as PWTimeoutError

logger = logging.getLogger(__name__)

# このモジュール用の Blueprint（既存をそのまま維持）
bp = Blueprint("livedoor_signup", __name__, url_prefix="/livedoor-signup")

# ─────────────────────────────────────────────
# 補助ユーティリティ（既存ロジックをそのまま活かす）
# ─────────────────────────────────────────────
import re as _re
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

def _slugify_ascii(s: str) -> str:
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

import random, string
def generate_safe_id(n=10) -> str:
    chars = string.ascii_lowercase + string.digits + "_"
    first_char = random.choice(string.ascii_lowercase)
    rest = ''.join(random.choices(chars, k=n - 1))
    return first_char + rest

def generate_safe_password(n=12) -> str:
    chars = string.ascii_letters + string.digits + "-_%$#"
    while True:
        password = ''.join(random.choices(chars, k=n))
        if any(c in "-_%$#" for c in password):
            return password

# ─────────────────────────────────────────────
# 新：CAPTCHA準備（セッション確保＆画像保存）— 同期API
# ─────────────────────────────────────────────
CAPTCHA_DIR = Path("app/static/captchas")
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

def prepare_captcha(email_addr: str, livedoor_id: str, password: str) -> Tuple[str, str]:
    """
    LiveDoor 会員登録フォームに入力→送信→CAPTCHAが出たら要素スクショを保存。
    返り値: (session_id, captcha_image_path)
    """
    sid, page = pwctl.run(pwctl.create_session(provider="livedoor"))
    img_path = pwctl.run(_ld_prepare(page, email_addr, livedoor_id, password, sid))
    # 復旧用に storage_state を保存（ワーカー跨ぎ/復活にも強くする）
    pwctl.run(pwctl.save_storage_state(sid))
    return sid, img_path

async def _ld_prepare(page: Page, email_addr: str, livedoor_id: str, password: str, session_id: str) -> str:
    logger.info("[LD-Signup] goto register/input (sid=%s)", session_id)
    await page.goto("https://member.livedoor.com/register/input", wait_until="load")

    await page.fill('input[name="livedoor_id"]', livedoor_id)
    await page.fill('input[name="password"]', password)
    await page.fill('input[name="password2"]', password)
    await page.fill('input[name="email"]', email_addr)

    await page.click('input[type="submit"][value="ユーザー情報を登録"]')

    img = page.locator("#captcha-img")
    try:
        await img.wait_for(state="visible", timeout=20_000)
    except PWTimeoutError:
        # attached→visible 切替の遅延にも一応対応
        await img.wait_for(state="attached", timeout=5_000)

    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = CAPTCHA_DIR / f"captcha_{session_id}_{ts}.png"
    await img.screenshot(path=str(img_path))

    logger.info("[LD-Signup] CAPTCHA画像を %s に保存 (sid=%s)", img_path, session_id)
    await pwctl.set_step(session_id, "captcha_required")
    return str(img_path)

# ─────────────────────────────────────────────
# 新：CAPTCHA送信（同一セッションで継続）— 同期API
# ─────────────────────────────────────────────
def submit_captcha(session_id: str, captcha_text: str) -> bool:
    """
    CAPTCHA文字列を送信し、/register/done に到達したら True。
    以降（メール認証→APIキー取得）は本関数内の“差込点”であなたの既存処理を呼び出してください。
    """
    page = pwctl.run(pwctl.get_page(session_id))
    if page is None:
        # ページを落としてしまっても storage_state から復旧可
        page = pwctl.run(pwctl.revive(session_id))
        if page is None:
            raise RuntimeError(f"signup session not found (sid={session_id})")

    ok = pwctl.run(_ld_submit(page, captcha_text, session_id))
    return ok

async def _ld_submit(page: Page, captcha_text: str, session_id: str) -> bool:
    logger.info("[LD-Signup] submit captcha (sid=%s)", session_id)

    # livedoor の CAPTCHA 入力欄（名称が違う場合はここだけ調整）
    await page.fill('input[name="captcha"]', captcha_text.replace(" ", "").replace("　", ""))

    # 送信（valueやidが変わっても拾えるよう汎用セレクタ）
    await page.click('input[type="submit"]')

    try:
        await page.wait_for_url("**/register/done", timeout=30_000)
    except PWTimeoutError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fail_png = CAPTCHA_DIR / f"failed_after_captcha_{session_id}_{ts}.png"
        try:
            await page.screenshot(path=str(fail_png), full_page=True)
        except Exception:
            pass
        logger.error("[LD-Signup] /register/done へ遷移せず（sid=%s）。スクショ: %s", session_id, fail_png)
        return False

    await pwctl.set_step(session_id, "captcha_submitted")
    logger.info("[LD-Signup] reached /register/done (sid=%s)", session_id)

    # ─────────────────────────────────────────
    # 差込点①：メール認証リンクを取得して開く（あなたの既存実装を呼ぶ）
    # 例:
    # from app.services.mail_utils.mail_tm import poll_latest_link_tm_async
    # activation_url = await poll_latest_link_tm_async(token, timeout_sec=120)
    # if not activation_url: return False
    # await page.goto(activation_url, wait_until="load")
    # await pwctl.set_step(session_id, "email_verified")
    # ─────────────────────────────────────────

    # ─────────────────────────────────────────
    # 差込点②：AtomPub/APIキー取得～DB保存（既存 recover を利用）
    # 例:
    # from app.services.blog_signup.livedoor_atompub_recover import recover_atompub_key
    # result = await recover_atompub_key(page, nickname, email, password, site, desired_blog_id=None)
    # if not result.get("success"): return False
    # await pwctl.set_step(session_id, "api_key_ok")
    # ─────────────────────────────────────────

    return True

# ─────────────────────────────────────────────
# 以降：旧・補助関数（メールURL抽出、手動確認画面など）
# ─────────────────────────────────────────────
import re
def extract_verification_url(email_body: str) -> str | None:
    pattern = r"https://member\.livedoor\.com/verify/[a-zA-Z0-9]+"
    m = re.search(pattern, email_body)
    return m.group(0) if m else None

import json, os
TEMP_DIR = "/tmp/livedoor_tasks"
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
    CAPTCHA後、認証リンクをユーザーに手動で表示する画面（既存フローを維持）。
    """
    from app.services.mail_utils.mail_tm import poll_latest_link_tm_async as poll_latest_link_gw
    email_body = poll_latest_link_gw(task_id=task_id, max_attempts=30, interval=5)

    if email_body:
        verification_url = extract_verification_url(email_body)
        if verification_url:
            return render_template("confirm_email.html", verification_url=verification_url)
        else:
            flash("認証リンクが見つかりませんでした", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("認証メールを取得できませんでした", "danger")
        return redirect(url_for('dashboard'))
