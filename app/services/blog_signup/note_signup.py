# -*- coding: utf-8 -*-
"""
NOTE (note.com) アカウント自動登録
playwright==1.53.0 で動作確認
"""

import logging, random, time
from playwright.sync_api import (
    sync_playwright, TimeoutError as PWTimeout, Error as PWError
)

SIGNUP_URL = "https://note.com/signup?signup_type=email"   # ← 直接ランディングへ
__all__ = ["signup_note_account"]


def _rand_wait(a=0.8, b=1.6):
    time.sleep(random.uniform(a, b))


def signup_note_account(email: str, password: str) -> dict:
    """
    Note アカウントを新規登録する
    Returns
    -------
    {"ok": bool, "error": str | None}
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(
                locale="ja-JP",
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/125.0.0.0 Safari/537.36"),
            )
            page = ctx.new_page()

            # 1️⃣ ランディングを開く
            page.goto(SIGNUP_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # 2️⃣ 「メールで登録」ボタンを押して“本フォーム”へ遷移
            try:
                page.get_by_role("button", name="メールで登録").click(timeout=10_000)
            except PWTimeout:
                logging.error("[note_signup] 'メールで登録' ボタンが見つからない")
                browser.close()
                return {"ok": False, "error": "signup button not found"}

            _rand_wait()

            # 3️⃣ email / password を入力（iframe なし）
            email_sel = (
                'input[name="email"], input[type="email"], '
                'input[placeholder*="メールアドレス"], input[placeholder*="mail@example.com"]'
            )
            page.wait_for_selector(email_sel, timeout=15_000)
            page.fill(email_sel, email)
            _rand_wait()

            pass_sel = (
                'input[name="password"], input[type="password"], '
                'input[placeholder*="パスワード"]'
            )
            page.fill(pass_sel, password)
            _rand_wait()

            # 4️⃣ 「同意して登録」クリック
            page.get_by_role("button", name="同意して登録").click()
            _rand_wait()

            # 5️⃣ 完了ページの確認
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:                                     # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
