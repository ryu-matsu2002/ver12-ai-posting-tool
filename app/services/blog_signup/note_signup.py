# -*- coding: utf-8 -*-
"""
note.com アカウント【完全自動】新規登録
------------------------------------------------------------
* Playwright 1.53.0 で動作確認
* 1) mail.tm でワンタイムメール inbox 生成
* 2) note.com へメール/パスワード/ニックネーム入力
* 3) reCAPTCHA v3 スコアが十分なら "同意して登録" ボタンが enabled
* 4) メール受信 → 認証リンククリックで本登録完了
------------------------------------------------------------
返り値:
    {"ok": bool, "email": str | None, "password": str | None, "error": str | None}
"""

from __future__ import annotations

import logging
import random
import string
import time
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
    Page,
)

# ---- 内部 util ---------------------------------------------------------------
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link

LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_PATH   = "/signup/form"
COMPLETE_PATH = "/signup/complete"

__all__ = ["signup_note_account"]

# ============= Helper =========================================================


def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _wait(a: float = 0.6, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))


STEALTH_JS = """
// --- Playwright 簡易ステルス対策 -------------------------------------------
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.navigator.chrome = { runtime: {} };
Object.defineProperty(navigator, 'languages', { get: () => ['ja-JP', 'ja'] });
Object.defineProperty(navigator, 'plugins', {
  get: () => [1, 2, 3, 4, 5],
});
"""


# ============= main ==========================================================


def signup_note_account() -> dict:
    """
    Note アカウントを mail.tm 経由で完全自動登録する。
    戻り値: {"ok": bool, "email": str|None, "password": str|None, "error": str|None}
    """
    # ------------------------------------------------------------------ 0. 使い捨てメール生成
    email, jwt = create_inbox()       # {"address":..., "token":...}
    password   = _rand(12) + "!"      # Note 側パスワード
    nickname   = f"user-{_rand(6)}"

    logging.info("[note_signup] new inbox %s", email)

    try:
        with sync_playwright() as p:
            # ---------------------------------------------------------------- 1. ブラウザ起動 (ステルス系オプション)
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                    "--disable-extensions",
                ],
            )
            ctx = browser.new_context(
                locale="ja-JP",
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/125.0.0.0 Safari/537.36"),
            )
            page: Page = ctx.new_page()
            page.add_init_script(STEALTH_JS)

            # ---------------------------------------------------------------- 2. ランディング → フォーム
            logging.info("▶ goto: %s", LANDING_URL)
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")
            _wait()

            # 「メールで登録」クリック（※たまに直でフォームになる）
            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
            except PWTimeout:
                pass  # ボタンが無い＝すでにフォーム

            # URL が /signup/form … に変わるまで待つ
            page.wait_for_url(f"**{FORM_PATH}**", timeout=15_000)
            logging.info("▶ formURL: %s", page.url)
            _wait()

            # ---------------------------------------------------------------- 3. フォーム入力
            page.fill('input[type="email"]', email, timeout=10_000)
            _wait()
            page.fill('input[type="password"]', password)
            _wait()
            if page.locator('input[name="nickname"]').count():
                page.fill('input[name="nickname"]', nickname)
                _wait()

            # 規約チェックボックス (あれば)
            cb = page.locator('input[type="checkbox"]')
            if cb.count():
                cb.check()

            # ---------------------------------------------------------------- 4. "同意して登録" ボタン待機 (最大30s)
            submit_btn = page.locator('button[type="submit"]')
            for _ in range(60):             # 0.5s × 60 = 30秒
                try:
                    if submit_btn.is_enabled():
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            else:
                logging.error("[note_signup] submit button never enabled")
                return {
                    "ok": False,
                    "email": email,
                    "password": password,
                    "error": "submit button never enabled",
                }

            submit_btn.click()
            logging.info("▶ submit clicked")

            # ---------------------------------------------------------------- 5. 完了ページ遷移
            page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)
            logging.info("▶ signup complete page reached")

            browser.close()

        # ------------------------------------------------------------------ 6. メールをポーリング → 認証リンク
        verify_link = poll_latest_link(jwt, sender_like="@note.com")
        if not verify_link:
            return {
                "ok": False,
                "email": email,
                "password": password,
                "error": "verification mail not found",
            }

        logging.info("▶ verify link: %s", verify_link)

        # Playwright でリンクを叩いて本登録
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(verify_link, timeout=30_000)
            page.wait_for_load_state("networkidle")
            browser.close()

        logging.info("[note_signup] SUCCESS: %s", email)
        return {"ok": True, "email": email, "password": password, "error": None}

    # ---------------- Error handling ----------------
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "email": email, "password": password, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "email": email, "password": password, "error": str(e)}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "email": email, "password": password, "error": str(e)}
