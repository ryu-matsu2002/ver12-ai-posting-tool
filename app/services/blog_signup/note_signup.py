# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（本番用・2025-07-06 FIX）
"""

from __future__ import annotations
import asyncio, json, logging, random, string, time
from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from .. import db
from app.models import ExternalBlogAccount

LANDING_URL   = "https://note.com/signup?signup_type=email"
FORM_PATH     = "/signup/form"
COMPLETE_PATH = "/signup/complete"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float = .6, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))

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
                slow_mo=200,
            )
            ctx  = await browser.new_context(locale="ja-JP")
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # 1) サインアップ TOP
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("domcontentloaded")

            # 2) フォーム or ボタン判定
            email_sel = "input[type='email'], input[name='email']"
            pwd_sel   = "input[type='password'], input[name='password']"

            if not await page.locator(email_sel).first.wait_for(state="attached", timeout=4_000):
                # フォームが無い → 「メールで登録」クリック
                await page.locator("text=メールで登録").first.click()

                # iframe 内のフォームを取得
                frame_email = page.frame_locator("iframe").locator(email_sel).first
                frame_pwd   = page.frame_locator("iframe").locator(pwd_sel).first
                await frame_email.wait_for(state="visible", timeout=20_000)

                fill_email = frame_email
                fill_pwd   = frame_pwd
            else:
                # 直接フォームがあるパターン
                fill_email = page.locator(email_sel).first
                fill_pwd   = page.locator(pwd_sel).first

            # 3) 入力
            await fill_email.fill(account.email)
            await fill_pwd.fill(account.password)

            # 4) ボタン活性化 → クリック
            await page.wait_for_selector(
                "[data-testid='signup-submit']:not([disabled])",
                timeout=30_000
            )
            await page.locator("[data-testid='signup-submit']").click()
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)

            # 5) storage_state 保存
            state_dir  = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # 6) DB 更新
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()

        logging.info("[note_signup] ✅ SUCCESS id=%s", account.id)
        return {"ok": True}

    except Exception as e:                        # noqa: BLE001
        logging.error("[note_signup] ❌ FAILED id=%s %s", account.id, e)
        return {"ok": False, "error": str(e)}

__all__ = ["signup_note_account"]
