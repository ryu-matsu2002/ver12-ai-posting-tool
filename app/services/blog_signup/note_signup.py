# -*- coding: utf-8 -*-
"""
NOTE (note.com) アカウント自動登録
playwright==1.53.0
"""

import logging, random, time
from playwright.sync_api import (
    sync_playwright, TimeoutError as PWTimeout, Error as PWError
)

# ① ランディング（ボタン付き）
LANDING_URL = "https://note.com/signup?signup_type=email"
# ② ボタンを押した後に遷移する“本フォーム”URL
FORM_URL    = "https://note.com/signup/form?redirectPath=%2Fsignup"

__all__ = ["signup_note_account"]


def _rand_wait(a=0.8, b=1.6):
    time.sleep(random.uniform(a, b))


def signup_note_account(email: str, password: str) -> dict:
    """
    Note アカウントを新規登録
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

            # ------------------------------------------------------------------
            # 1️⃣ ランディング
            # ------------------------------------------------------------------
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # “メールで登録” ボタン（<button> or <a> のテキスト一致）を探す
            clicked = False
            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                clicked = True
            except PWTimeout:
                # ボタンが無い → フォーム URL へダイレクト遷移
                page.goto(FORM_URL, timeout=15_000)

            if clicked:
                # 自動遷移が終わるまで待つ
                page.wait_for_url("**/signup/form**", timeout=15_000)

            _rand_wait()

            # ------------------------------------------------------------------
            # 2️⃣ email / password 入力
            # ------------------------------------------------------------------
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

            # 「同意して登録」
            page.locator("text=同意して登録").click()
            _rand_wait()

            # ------------------------------------------------------------------
            # 3️⃣ 完了確認
            # ------------------------------------------------------------------
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    # ---------------- エラーハンドリング ----------------
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:                                    # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
