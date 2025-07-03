# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録
playwright==1.53.0 で検証
"""

import logging, random, time
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
)

LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_URL    = "https://note.com/signup/form?redirectPath=%2Fsignup"
__all__     = ["signup_note_account"]

def _wait(a: float = .6, b: float = 1.2) -> None:
    time.sleep(random.uniform(a, b))

# ──────────────────────────────────────────────
def signup_note_account(email: str, password: str) -> dict:
    """メール & パスワードで Note にサインアップ"""
    try:
        with sync_playwright() as p:
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

            # 1️⃣ ランディング → 登録フォーム
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")
            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                page.wait_for_url("**/signup/form**", timeout=15_000)
            except PWTimeout:
                page.goto(FORM_URL, timeout=15_000)
            _wait()

            # 2️⃣ 入力欄
            email_in = (
                'input[name="email"], input[type="email"], '
                'input[placeholder*="メールアドレス"]'
            )
            pass_in = (
                'input[name="password"], input[type="password"], '
                'input[placeholder*="パスワード"]'
            )
            page.locator(email_in).first.fill(email, timeout=15_000)
            _wait()
            page.locator(pass_in).first.fill(password, timeout=15_000)
            _wait()

            # 3️⃣ ボタン enable 待ち → クリック
            signup_btn = page.locator('button:has-text("同意して登録")')
            signup_btn.wait_for(state="enabled", timeout=15_000)
            signup_btn.click()

            # 4️⃣ 完了ページ
            page.wait_for_url("**/signup/complete**", timeout=60_000)

            browser.close()
            return {"ok": True, "error": None}

    # ─── エラーハンドリング ──────────────────────
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
