# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（本番用・2025-07-07 FIX v2）
====================================================================
signup_note_account(account: ExternalBlogAccount) -> {"ok":True}|{"ok":False,…}

◉ 修正点
────────────────────────────────────────────
1. <input type="email"> が出て来るまで **最大 10 秒** 待機。
   出て来なければ必ず「メールで登録」ボタン → iframe 内フォームを待つフォールバック。
2. ログを詳細化してデバッグしやすく。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from .. import db
from app.models import ExternalBlogAccount

# ─────────────────────────────────────────────
LANDING_URL   = "https://note.com/signup?signup_type=email"
FORM_PATH     = "/signup/form"
COMPLETE_PATH = "/signup/complete"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# ---------- helpers ----------------------------------------------------------
def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float = .6, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))

# ---------- main -------------------------------------------------------------
async def signup_note_account(account: ExternalBlogAccount) -> Dict[str, str | bool]:
    """
    Note 新規アカウントを Playwright で登録し、storage_state を保存。
    成功: {"ok": True}
    失敗: {"ok": False, "error": "..."}
    """
    if not (account.email and account.password):
        return {"ok": False, "error": "email/password not set"}

    nickname = account.nickname or f"user-{_rand(6)}"
    logging.info("[note_signup] ▶ START id=%s mail=%s", account.id, account.email)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                ],
                slow_mo=200,
            )
            ctx = await browser.new_context(locale="ja-JP")
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # 1) signup ランディング
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("domcontentloaded")

            email_sel = "input[type='email'], input[name='email']"
            pwd_sel   = "input[type='password'], input[name='password']"

            # 2) 直接フォームを 10 秒待つ ─────────────────
            logging.info("[note_signup] waiting email form (≤10s)…")
            try:
                await page.locator(email_sel).first.wait_for(
                    state="attached", timeout=10_000
                )
                fill_email = page.locator(email_sel).first
                fill_pwd   = page.locator(pwd_sel).first
            except PWTimeout:
                # 3) 無ければ「メールで登録」→ iframe 内フォーム
                logging.info("[note_signup] fallback ⇒ click 'メールで登録'")
                await page.locator("text=メールで登録").first.click()
                frame = page.frame_locator("iframe").nth(0)

                await frame.locator(email_sel).first.wait_for(
                    state="visible", timeout=20_000
                )
                fill_email = frame.locator(email_sel).first
                fill_pwd   = frame.locator(pwd_sel).first

            # 4) 入力
            await fill_email.fill(account.email)
            await fill_pwd.fill(account.password)

            # 5) サインアップボタン活性化を待機 → クリック
            await page.wait_for_selector(
                '[data-testid="signup-submit"]:not([disabled])', timeout=30_000
            )
            await page.locator('[data-testid="signup-submit"]').click()

            # 6) 完了ページ遷移確認
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)

            # 7) storage_state 保存
            state_dir  = Path("storage_states")
            state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # 8) DB 更新
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()

        logging.info("[note_signup] ✅ SUCCESS id=%s", account.id)
        return {"ok": True}

    except Exception as e:  # noqa: BLE001
        logging.error("[note_signup] ❌ FAILED id=%s %s", account.id, e)
        return {"ok": False, "error": str(e)}


__all__ = ["signup_note_account"]
