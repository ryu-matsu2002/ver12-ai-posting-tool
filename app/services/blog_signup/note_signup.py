# -*- coding: utf-8 -*-
"""
note.com アカウント自動登録（2025-07 修正版）
--------------------------------------------------
● Playwright 1.53.0 で動作確認
● 手順
    1. 「メールで登録」をクリックして /signup/form に遷移
    2. ランダムメール（Mail.tm）と強力パスワードを入力
    3. ニックネーム入力・利用規約チェック
    4. 「同意して登録」ボタンが有効になったらクリック
    5. /signup/complete に遷移すれば成功
"""

import logging
import random
import string
import time
from typing import Tuple

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Error as PWError,
    Page,
)

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_URL = "https://note.com/signup/form"
WAIT = 0.8, 1.6        # ランダム待機（sec）
TIMEOUT = 30_000       # 最大待機（ms）

__all__ = ["signup_note_account"]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def _rand_wait(a: float = WAIT[0], b: float = WAIT[1]) -> None:
    time.sleep(random.uniform(a, b))


def _random_password(length: int = 12) -> str:
    """強力パスワード生成"""
    chars = string.ascii_letters + string.digits + "!@#$%&*?"
    return "".join(random.choice(chars) for _ in range(length))


def _create_random_email() -> Tuple[str, str]:
    """簡易ランダムメール（Mail.tm を使用。失敗時は例外送出）"""
    import requests, json, uuid, secrets

    session = requests.Session()
    # 1️⃣ ドメイン取得
    domain = session.get("https://api.mail.tm/domains").json()["hydra:member"][0]["domain"]
    # 2️⃣ アカウント作成
    user = f"{uuid.uuid4().hex[:10]}@{domain}"
    pw = secrets.token_urlsafe(12)
    resp = session.post(
        "https://api.mail.tm/accounts",
        json={"address": user, "password": pw},
        timeout=10,
    )
    resp.raise_for_status()
    token = session.post(
        "https://api.mail.tm/token",
        json={"address": user, "password": pw},
        timeout=10,
    ).json()["token"]
    return user, token  # token は未使用だが戻しておく


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def signup_note_account() -> dict:
    """
    Returns
    -------
    dict
        {"ok": bool, "email": str | None, "password": str | None, "error": str | None}
    """
    # ① ランダムメール & PW 生成
    try:
        email, _ = _create_random_email()
    except Exception as e:
        return {"ok": False, "email": None, "password": None, "error": f"mail.tm error: {e}"}

    password = _random_password()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            ctx = browser.new_context(locale="ja-JP")
            page: Page = ctx.new_page()

            # ─────────────────────────────────────────────
            # STEP 0: ランディング
            # ─────────────────────────────────────────────
            page.goto(LANDING_URL, timeout=TIMEOUT)
            print("▶ goto:", page.url)

            # 「メールで登録」ボタン押下
            register_btn = page.locator("text=メールで登録").first
            register_btn.wait_for(state="visible", timeout=10_000)
            register_btn.click()
            print("▶ メールで登録クリック")

            # `/signup/form` に遷移
            page.wait_for_url("**/signup/form**", timeout=TIMEOUT)
            print("▶ formURL:", page.url)

            # ─────────────────────────────────────────────
            # STEP 1: フォーム入力
            # ─────────────────────────────────────────────
            _rand_wait()
            page.fill('input[type="email"]', email)
            _rand_wait()
            page.fill('input[type="password"]', password)

            # ニックネーム
            if page.locator('input[name="nickname"]').count():
                page.fill('input[name="nickname"]', f"user_{random.randint(1000,9999)}")

            # 規約チェック
            cb = page.locator('input[type="checkbox"]')
            if cb.count():
                cb.check()

            # ─────────────────────────────────────────────
            # STEP 2: ボタン有効化待ち（手動ループに変更）
            # ─────────────────────────────────────────────
            submit_btn = page.locator('button[type="submit"]')
            for _ in range(30):  # 最大15秒間試行
                if submit_btn.is_enabled():
                    break
                time.sleep(0.5)
            else:
                logging.error("[note_signup] signup button never enabled (captcha?)")
                return {"ok": False, "email": None, "password": None, "error": "signup button never enabled"}

            submit_btn.click()
            print("▶ submit click")


            # ─────────────────────────────────────────────
            # STEP 3: 完了ページ確認
            # ─────────────────────────────────────────────
            page.wait_for_url("**/signup/complete**", timeout=TIMEOUT)
            print("✅ signup complete:", page.url)

            browser.close()
            return {"ok": True, "email": email, "password": password, "error": None}

    # ─────────── 例外処理 ───────────
    except PWTimeout as e:
        logging.error("[note_signup] Timeout: %s", e)
        return {"ok": False, "email": None, "password": None, "error": f"Timeout: {e}"}

    except PWError as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "email": None, "password": None, "error": str(e)}

    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "email": None, "password": None, "error": str(e)}
