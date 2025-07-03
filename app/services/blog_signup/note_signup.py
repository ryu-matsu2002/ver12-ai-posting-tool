# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録
playwright==1.53.0 で検証
"""

import logging
import random
import time

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
)

LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_URL    = "https://note.com/signup/form?redirectPath=%2Fsignup"
__all__     = ["signup_note_account"]


# ──────────────────────────────────────────────
def _wait(a: float = 0.6, b: float = 1.2) -> None:
    """人間らしいランダム wait"""
    time.sleep(random.uniform(a, b))
# ──────────────────────────────────────────────


def signup_note_account(email: str, password: str) -> dict:
    """
    Note にメール登録 → 完了ページ到達を自動化

    Returns
    -------
    dict
      {"ok": True , "error": None}   成功
      {"ok": False, "error": "..."}  失敗
    """
    try:
        with sync_playwright() as p:
            # 0️⃣ Browser
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(
                locale="ja-JP",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            )
            page = ctx.new_page()

            # 1️⃣ landing → フォーム
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                page.wait_for_url("**/signup/form**", timeout=15_000)
            except PWTimeout:
                page.goto(FORM_URL, timeout=15_000)

            _wait()

            # 2️⃣ 入力
            email_in = (
                'input[name="email"], input[type="email"], '
                'input[placeholder*="メールアドレス"]'
            )
            pass_in = (
                'input[name="password"], input[type="password"], '
                'input[placeholder*="パスワード"]'
            )

            page.wait_for_selector(email_in, timeout=15_000)
            page.click(email_in)
            page.keyboard.type(email, delay=50)
            page.dispatch_event(email_in, "blur")
            _wait()

            page.click(pass_in)
            page.keyboard.type(password, delay=50)
            page.dispatch_event(pass_in, "blur")
            _wait()

            # 3️⃣ 「同意して登録」ボタンが enabled になるまで待つ
            signup_btn = 'button:has-text("同意して登録")'

            page.wait_for_function(
                """selector => {
                     const el = document.querySelector(selector);
                     return el && !el.disabled;
                 }""",
                arg=signup_btn,          # ← ★ キーワード引数で渡す
                timeout=15_000,
            )

            page.locator(signup_btn).click(force=True)

            # 4️⃣ 完了ページ確認
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    # ── error handling ─────────────────────────
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:                      # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
