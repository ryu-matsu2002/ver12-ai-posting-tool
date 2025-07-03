# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録
playwright==1.53.0 で動作確認
"""

import logging
import random
import time
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
)

# ────────────────────────────────────────────────────────────
LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_URL    = "https://note.com/signup/form?redirectPath=%2Fsignup"
__all__     = ["signup_note_account"]
# ────────────────────────────────────────────────────────────


def _wait(a: float = 0.6, b: float = 1.2) -> None:
    """人間っぽいランダム wait"""
    time.sleep(random.uniform(a, b))


def signup_note_account(email: str, password: str) -> dict:
    """
    Note にメール＆パスワードでサインアップする。

    Returns
    -------
    {"ok": True,  "error": None}            成功
    {"ok": False, "error": "<msg>"}         失敗
    """
    try:
        with sync_playwright() as p:
            # ── 0. ブラウザ起動 ───────────────────────────────
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

            # ── 1. ランディング → フォーム ───────────────────
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # 「メールで登録」ボタンがあればクリック、なければ直接フォーム URL へ
            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                page.wait_for_url("**/signup/form**", timeout=15_000)
            except PWTimeout:
                page.goto(FORM_URL, timeout=15_000)

            _wait()

            # ── 2. メール & パスワード入力 ───────────────────
            email_in = (
                'input[name="email"], input[type="email"], '
                'input[placeholder*="メールアドレス"]'
            )
            pass_in = (
                'input[name="password"], input[type="password"], '
                'input[placeholder*="パスワード"]'
            )

            # メール
            page.wait_for_selector(email_in, timeout=15_000)
            page.click(email_in)
            page.keyboard.type(email, delay=50)
            #   blur を飛ばしてバリデーションを強制
            page.dispatch_event(email_in, "blur")
            _wait()

            # パスワード
            page.click(pass_in)
            page.keyboard.type(password, delay=50)
            page.dispatch_event(pass_in, "blur")
            _wait()

            # ── 3. 「同意して登録」ボタンを待つ → クリック ───
            btn_all = 'button:has-text("同意して登録")'
            # JS で “enabled になるまで” ポーリング
            page.wait_for_function(
                """sel => {
                     const el = document.querySelector(sel);
                     return el && !el.disabled;
                   }""",
                btn_all,
                timeout=15_000,
            )

            page.locator(btn_all).click(force=True)

            # ── 4. 完了ページへ ────────────────────────────
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    # ── エラーハンドリング ──────────────────────────────────
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:                             # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
