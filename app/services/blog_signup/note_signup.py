# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録
Mail.tm で取得した使い捨てメールを使って完全自動サインアップ
playwright==1.53.0
"""

import logging, random, string, time
from playwright.sync_api import (
    sync_playwright, TimeoutError as PWTimeout, Error as PWError
)
from app.services.mail_utils.mail_tm import create_inbox, wait_link

LANDING = "https://note.com/signup?signup_type=email"
FORM    = "https://note.com/signup/form?redirectPath=%2Fsignup"

__all__ = ["signup_note_account"]


def _rand_pw(n=10):
    return (
        random.choice(string.ascii_uppercase)
        + random.choice(string.ascii_lowercase)
        + random.choice("0123456789")
        + random.choice("!#@$")
        + "".join(random.choices(string.ascii_letters + string.digits, k=n - 4))
    )


def _w(a=0.8, b=1.5):
    time.sleep(random.uniform(a, b))


def signup_note_account() -> dict:
    """
    完全自動で note アカウントを作成
    Returns
    -------
    dict = {
        "ok": bool,
        "email": str|None,
        "password": str|None,
        "error": str|None,
    }
    """
    email, jwt = create_inbox()            # ← Mail.tm でメール確保
    password   = _rand_pw()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(locale="ja-JP")
            page = ctx.new_page()

            # ───────────────────────────────────────────────────────── landing
            page.goto(LANDING, timeout=30_000)
            page.locator("text=メールで登録").first.click()
            page.wait_for_url("**/signup/form**", timeout=15_000)

            # ────────────────────────────────────────────────────────── inputs
            page.fill('input[type="email"]', email)
            _w()
            page.fill('input[type="password"]', password)
            _w()

            # ニックネーム欄がある場合
            if page.locator('input[name="nickname"]').count():
                page.fill('input[name="nickname"]', "user" + email.split("@")[0])

            # 規約チェックボックス (存在時のみ)
            cb = page.locator('input[type="checkbox"]')
            if cb.count():
                cb.check()

            # ボタンが有効になるまで待機（ ← 修正ポイント ）
            page.wait_for_function(
                "() => {"
                "  const btn = document.querySelector('button[type=\"submit\"]');"
                "  return btn && !btn.disabled;"
                "}",                # expression
                timeout=15_000      # ← ★キーワード引数で渡す
            )

            page.locator('button[type="submit"]').click()
            _w()

            # ───────────────────────────────────────── メールの確認リンク
            verify_url = wait_link(jwt, timeout_sec=120)
            if not verify_url:
                raise RuntimeError("verification mail not received")

            page.goto(verify_url, timeout=30_000)
            page.wait_for_url("**/signup/complete**", timeout=30_000)

            browser.close()
            return {"ok": True, "email": email, "password": password, "error": None}

    # ───────────── error handling
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "email": None, "password": None, "error": f"Timeout: {e}"}

    except (PWError, Exception) as e:        # noqa: BLE001
        logging.error("[note_signup] %s", e, exc_info=True)
        return {"ok": False, "email": None, "password": None, "error": str(e)}
