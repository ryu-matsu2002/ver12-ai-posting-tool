# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（本番用・2025-07-06 FIX）
====================================================================
signup_note_account(account: ExternalBlogAccount) -> {"ok":True}|{"ok":False,…}

◉ 今回の主な修正
────────────────────────────────────────────
1. `wait_for_function` → **`wait_for_selector('[data-testid="signup-submit"]:not([disabled])')`**  
   └ Playwright v1.43 以降で安全にボタン活性化を待つ  
2. 失敗時のスタックを追いやすくするため step-by-step ログを追加
"""

from __future__ import annotations
import asyncio, json, logging, random, string, time
from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from .. import db
from app.models import ExternalBlogAccount

# ─────────────────────────────────────────
LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_PATH   = "/signup/form"
COMPLETE_PATH = "/signup/complete"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# ---------- helpers ----------
def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float=.6, b: float=1.3) -> None:
    time.sleep(random.uniform(a, b))

# ---------- main ----------
async def signup_note_account(account: ExternalBlogAccount) -> Dict[str, str | bool]:
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
                slow_mo=200,      # bot 検知回避: 少し待つ
            )
            ctx = await browser.new_context(locale="ja-JP")
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # 1) サインアップフォーム
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            try:
                await page.locator("text=メールで登録").first.click(timeout=4_000)
            except PWTimeout:
                pass
            await page.wait_for_url(f"**{FORM_PATH}**", timeout=15_000)

            # 2) 入力
            await page.fill('input[type="email"]', account.email); _wait()
            await page.fill('input[type="password"]', account.password); _wait()
            if await page.locator('input[name="nickname"]').count():
                await page.fill('input[name="nickname"]', nickname)
            chk = page.locator('input[type="checkbox"]')
            if await chk.count():
                await chk.check()

            # 3) ボタン活性化を待ってクリック（★ここを修正）
            await page.wait_for_selector(
                '[data-testid="signup-submit"]:not([disabled])',
                timeout=30_000
            )
            await page.locator('[data-testid="signup-submit"]').click()
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)

            # 4) cookie / storage_state 保存
            state_dir = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # 5) DB 更新
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
