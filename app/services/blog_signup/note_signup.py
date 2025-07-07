# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（iframe 対応 完全版）
================================================================
signup_note_account(account: ExternalBlogAccount) -> dict
  • ExternalBlogAccount を受け取り、Playwright でサインアップ
  • storage_state.json を保存し DB を更新
  • 成功: {"ok": True}
    失敗: {"ok": False, "error": "..."}   ※失敗時は PNG を保存
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

# ──────────────────────────────────────────────
LANDING_URL = "https://note.com/signup?signup_type=email"
COMPLETE_PATH = "/signup/complete"

SEL_EMAIL = "input[type='email'], input[name='email']"
SEL_PWD   = "input[type='password'], input[name='password']"
SEL_BTN   = "[data-testid='signup-submit'], button[type='submit']"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# ──────────────────────────────────────────────
def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float = 0.6, b: float = 1.2) -> None:
    time.sleep(random.uniform(a, b))

# ──────────────────────────────────────────────
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
                slow_mo=150,
            )
            ctx = await browser.new_context(locale="ja-JP")
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # 1️⃣ LP へ遷移
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("domcontentloaded")

            # 2️⃣ email/password フィールド探索（ページ + すべての iframe）
            target_email = target_pwd = None
            form_ctx = None                     # ボタン等を探すフレーム
            for f in [page, *page.frames]:
                email_el = f.locator(SEL_EMAIL).first
                if await email_el.count():
                    pwd_el = f.locator(SEL_PWD).first
                    if await pwd_el.count():
                        target_email, target_pwd = email_el, pwd_el
                        form_ctx = f
                        logging.info("[note_signup] ✓ form found in %s", "iframe" if f is not page else "page")
                        break

            if target_email is None:
                raise RuntimeError("e-mail form iframe not found")

            # 3️⃣ 入力
            await target_email.fill(account.email)
            await target_pwd.fill(account.password)

            # 4️⃣ submit ボタンクリック（iframe 内でも OK）
            submit_btn = form_ctx.locator(SEL_BTN).first
            await submit_btn.wait_for(state="visible", timeout=60_000)
            await submit_btn.click()

            # 5️⃣ 完了ページ待機
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=60_000)

            # 6️⃣ storage_state 保存
            state_dir = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # 7️⃣ DB 更新
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()
            logging.info("[note_signup] ✅ SUCCESS id=%s", account.id)
            return {"ok": True}

    except Exception as e:                       # すべて握って PNG 保存
        try:
            err_dir = Path("storage_states"); err_dir.mkdir(exist_ok=True)
            png = err_dir / f"signup_fail_{account.id}.png"
            if 'page' in locals():
                await page.screenshot(path=str(png))
        except Exception:                        # screenshot 失敗は無視
            pass

        logging.error("[note_signup] ❌ FAILED id=%s %s", account.id, e)
        return {"ok": False, "error": str(e)}

__all__ = ["signup_note_account"]
