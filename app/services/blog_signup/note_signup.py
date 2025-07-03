"""
NOTE の会員登録を Playwright で自動化
playwright==1.53.0 で動作確認
"""

from __future__ import annotations

import logging
import random
import time
from typing import List, Dict, Any

from playwright.sync_api import (
    sync_playwright,
    Page,
    TimeoutError as PWTimeout,
)

# ────────────────────────────────────────────────────────────
# 「?signup_type=email」を付けると最初からメール入力画面が開く
SIGNUP_URL = "https://note.com/signup?signup_type=email"
__all__ = ["signup_note_account"]
# ────────────────────────────────────────────────────────────


# ------------------------- 共通ユーティリティ --------------------------
def _random_wait(a: float = 0.8, b: float = 1.6) -> None:
    """人間らしい微小待機を挟む"""
    time.sleep(random.uniform(a, b))


def _wait_first(page: Page, selectors: List[str], timeout: int = 30_000):
    """
    selectors のうち *先に見つかった* Locator を返す。
    1 つも見つからなければ TimeoutError。
    """
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        for sel in selectors:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                return loc.first
        time.sleep(0.25)
    raise PWTimeout(f"none of selectors visible within {timeout} ms: {selectors}")


# ------------------------------- 本 体 ---------------------------------
def signup_note_account(email: str, password: str) -> Dict[str, Any]:
    """
    Note アカウントを新規登録する
    Returns
    -------
    {"ok": True,  "error": None }  … 成功
    {"ok": False, "error": str }  … 失敗内容
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,  # デバッグ時は False
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(locale="ja-JP")
            page: Page = ctx.new_page()

            # 1️⃣ メールサインアップ画面へ
            page.goto(SIGNUP_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            # （UI 変更対策）もしまだ選択メニューだったら「メールアドレスで登録」をクリック
            try:
                _wait_first(
                    page,
                    [
                        'button:has-text("メールアドレスで登録")',
                        'button:has-text("メールアドレスで会員登録")',
                    ],
                    timeout=3_000,
                ).click()
                page.wait_for_load_state("networkidle")
            except PWTimeout:
                # 既にメール入力フォームの場合はスルー
                pass

            # 2️⃣ 入力フォーム要素を取得
            email_box = _wait_first(
                page,
                [
                    'input[type="email"]',
                    'input[name="email"]',
                    'input[placeholder*="mail@example.com"]',
                    'input[placeholder*="メールアドレス"]',
                    "input",  # 最終手段：最初の <input>
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
                    "input[type='text'] + input",  # email の次の input を想定
                ],
            )
            pwd_box.fill(password)
            _random_wait()

            # 3️⃣ 「同意して登録」ボタン
            _wait_first(
                page,
                [
                    'button:has-text("同意して登録")',
                    'button:has-text("Register")',
                    'button[type="submit"]',
                ]
            ).click()

            # 4️⃣ 完了 URL へ遷移するまで最大 60 秒待機
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
