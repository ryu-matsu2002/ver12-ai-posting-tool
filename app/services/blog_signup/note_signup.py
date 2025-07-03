"""
NOTE の会員登録を Playwright で自動化するユーティリティ
playwright==1.53.0 で動作確認
"""

from __future__ import annotations

import logging
import random
import string
import time
from typing import List, Dict, Any

from playwright.sync_api import (
    sync_playwright,
    Page,
    TimeoutError as PWTimeout,
)

# ----------------------------------------------------------------------
SIGNUP_URL = "https://note.com/signup"          # エントリーポイント
__all__ = ["signup_note_account"]               # 外部公開シンボル
# ----------------------------------------------------------------------


# ------------------------- 共通ユーティリティ --------------------------
def _random_wait(a: float = 0.8, b: float = 1.6) -> None:
    """人間らしい待ちを挟む"""
    time.sleep(random.uniform(a, b))


def _wait_first(page: Page, selectors: List[str], timeout: int = 30_000):
    """
    与えられた selectors のうち *最初に見つかった* Locator を返す。
    すべて見つからなければ Playwright の TimeoutError を投げる。
    """
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        for sel in selectors:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc
        time.sleep(0.25)
    raise PWTimeout(f"none of selectors found within {timeout} ms: {selectors}")


# ---------------------------- 本 体 -----------------------------------
def signup_note_account(email: str, password: str) -> Dict[str, Any]:
    """
    Note アカウントを新規登録する。

    Parameters
    ----------
    email : str
        登録するメールアドレス
    password : str
        登録するパスワード（8 文字以上の半角英数記号）

    Returns
    -------
    dict
        {"ok": True, "error": None}  … 成功
        {"ok": False, "error": "…"} … 失敗
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,                     # =False で動きを目視確認可
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            ctx = browser.new_context(locale="ja-JP")
            page: Page = ctx.new_page()

            # 1️⃣ サインアップページへ遷移
            page.goto(SIGNUP_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # 2️⃣ メール - パスワード入力
            email_box = _wait_first(
                page,
                [
                    'input[type="email"]',
                    'input[name="email"]',
                    'input[placeholder*="mail@example.com"]',
                    'input[placeholder*="メールアドレス"]',
                ],
            )
            email_box.fill(email)
            _random_wait()

            pwd_box = _wait_first(
                page,
                [
                    'input[type="password"]',
                    'input[name="password"]',
                    'input[placeholder*="パスワード"]',
                ],
            )
            pwd_box.fill(password)
            _random_wait()

            # 3️⃣ 「同意して登録」ボタンをクリック
            _wait_first(
                page,
                [
                    'button:has-text("同意して登録")',
                    'button:has-text("Register")',
                ]
            ).click()

            # 4️⃣ 完了ページへ遷移するまで最大 60 秒待機
            page.wait_for_url("**/signup/complete**", timeout=60_000)

            browser.close()
            return {"ok": True, "error": None}

    # ---------- エラー処理 ----------
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
