# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録スクリプト
動作確認環境: playwright==1.53.0

外部からは signup_note_account(email, password) を呼び出すだけで OK。
返り値: {"ok": True/False, "error": str | None}
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, Any

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
    Page,
)

# --------------------------------------------------------------------
# 定数
# --------------------------------------------------------------------
LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_URL = "https://note.com/signup/form?redirectPath=%2Fsignup"

__all__ = ["signup_note_account"]


def _wait(a: float = 0.6, b: float = 1.2) -> None:
    """入力後などに少しランダム待機（簡易人間っぽさ対策）"""
    time.sleep(random.uniform(a, b))


# --------------------------------------------------------------------
# メイン関数
# --------------------------------------------------------------------
def signup_note_account(email: str, password: str) -> Dict[str, Any]:
    """
    Note にメール認証方式でサインアップを試みる。

    Returns
    -------
    dict : {"ok": bool, "error": str | None}
    """
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
            page: Page = ctx.new_page()

            # ────────────────────────────────────────────────
            # 1️⃣ ランディング → 「メールで登録」クリック
            # ────────────────────────────────────────────────
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")

            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
                page.wait_for_url("**/signup/form**", timeout=15_000)
            except PWTimeout:
                # ボタンが見つからないときは直接フォーム URL
                page.goto(FORM_URL, timeout=15_000)

            _wait()

            # ────────────────────────────────────────────────
            # 2️⃣ メール & パス & ニックネーム入力
            #    （.type を使い keyup でボタン活性化）
            # ────────────────────────────────────────────────
            # メール
            email_in = (
                'input[name="email"], input[type="email"], '
                'input[placeholder*="メールアドレス"]'
            )
            page.wait_for_selector(email_in, timeout=15_000)
            page.locator(email_in).first.click()
            page.keyboard.type(email, delay=40)
            _wait()

            # パスワード
            pass_in = (
                'input[name="password"], input[type="password"], '
                'input[placeholder*="パスワード"]'
            )
            page.locator(pass_in).first.click()
            page.keyboard.type(password, delay=40)
            _wait()

            # ✅ ニックネーム（必須） ← ここが今回の追加
            nick_in = (
                'input[name="name"], input[placeholder*="ニックネーム"], '
                'input[placeholder*="ユーザー名"]'
            )
            if page.locator(nick_in).count():
                nickname = f"user{int(time.time()) % 100000}"
                page.locator(nick_in).first.click()
                page.keyboard.type(nickname, delay=40)
                _wait()

            # ✅ 利用規約チェック ← ここが今回の追加
            ck_box = 'input[type="checkbox"]'
            if page.locator(ck_box).count():
                # 既にチェック済みなら skip
                if not page.locator(ck_box).first.is_checked():
                    page.locator(ck_box).first.check()
                    _wait()

            # ────────────────────────────────────────────────
            # 3️⃣ 「同意して登録」ボタンが enabled になるのを待つ
            # ────────────────────────────────────────────────
            btn_selector = 'button:has-text("同意して登録")'
            try:
                page.locator(btn_selector).first.wait_for(
                    state="visible", timeout=15_000
                )
                # disabled 属性が外れるまで最大 10 秒待機
                for _ in range(100):
                    if page.locator(btn_selector).first.is_enabled():
                        break
                    time.sleep(0.1)
                else:
                    raise PWTimeout("signup button never enabled")
            except PWTimeout as e:
                logging.error("[note_signup] Timeout: %s", e)
                browser.close()
                return {"ok": False, "error": str(e)}

            page.locator(btn_selector).first.click(force=True)

            # ────────────────────────────────────────────────
            # 4️⃣ 完了ページを待つ
            # ────────────────────────────────────────────────
            page.wait_for_url("**/signup/complete**", timeout=60_000)
            browser.close()
            return {"ok": True, "error": None}

    # ---------------------- 例外処理 ----------------------
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "error": str(e)}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "error": str(e)}
