# -*- coding: utf-8 -*-
"""
note.com アカウント完全自動登録（Mail.tm + Playwright）
"""

import logging, random, time, re
from playwright.sync_api import (
    sync_playwright, TimeoutError as PWTimeout, Error as PWError
)
from app.services.mail_utils.mail_tm import create_inbox, wait_link

__all__ = ["signup_note_account"]

LANDING = "https://note.com/signup?signup_type=email"
FORM_PAT = "**/signup/form**"
COMPLETE_PAT = "**/signup/complete**"

def _w(a=0.6, b=1.2):
    time.sleep(random.uniform(a, b))

# ---------------------------------------------------------------------

def signup_note_account(nickname_prefix: str = "user") -> dict:
    """
    戻り値: {"ok": bool, "email": str | None, "error": str | None}
    """
    # 1️⃣ 使い捨てメール作成
    email, jwt = create_inbox()
    password = f"Pwd{random.randint(10000,99999)}!"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            ctx = browser.new_context(
                locale="ja-JP",
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/125.0.0.0 Safari/537.36")
            )
            page = ctx.new_page()

            # 2️⃣ フォームへ遷移
            page.goto(LANDING, timeout=30_000)
            page.locator("text=メールで登録").first.click()
            page.wait_for_url(FORM_PAT, timeout=15_000)

            # 3️⃣ 入力（遅延タイプでreCAPTCHA対策）
            def slow_fill(sel, txt):
                page.click(sel)
                for ch in txt:
                    page.keyboard.type(ch, delay=random.randint(80, 140))
                _w()

            slow_fill('input[type="email"]', email)
            slow_fill('input[type="password"]', password)

            # optional nickname
            nick_sel = 'input[name="nickname"]'
            if page.locator(nick_sel).count():
                slow_fill(nick_sel, f"{nickname_prefix}{random.randint(1000,9999)}")

            # checkbox
            cb = page.locator('input[type="checkbox"]')
            if cb.count():
                cb.check()

            # ボタン enable 待機
            btn = page.locator('button:has-text("同意して登録")')
            page.wait_for_function(
                "(b)=>!b.disabled", btn, timeout=20_000
            )
            btn.click()
            _w()

            # 4️⃣ 認証メールを待機 → リンク GET
            link = wait_link(jwt, subject_kw="メールアドレスの確認")
            if not link:
                raise RuntimeError("verification mail timeout")

            # 5️⃣ リンクを開いて完了判定
            page.goto(link, timeout=30_000)
            page.wait_for_url(COMPLETE_PAT, timeout=30_000)

            browser.close()
            return {"ok": True, "email": email, "password": password, "error": None}

    # ---------- Error ----------
    except (PWTimeout, PWError, RuntimeError) as e:
        logging.error("[note_signup] %s", e)
        return {"ok": False, "email": None, "error": str(e)}
