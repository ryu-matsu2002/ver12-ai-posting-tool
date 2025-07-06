# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（本番用）
==============================================================
signup_note_account(account: ExternalBlogAccount) -> dict
  * ExternalBlogAccount を受け取り、Playwright でサインアップ
  * storage_state.json を保存し DB を更新
  * 戻り値: {"ok": True} もしくは {"ok": False, "error": "..."}
--------------------------------------------------------------
※ 旧 legacy_signup_note()（mail.tm を経由する検証向け）は
   2025-07-06 時点で未使用になったため完全に削除しました。
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

# ── Playwright ────────────────────────────────────────────────
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── プロジェクト内部 ──────────────────────────────────────────
from .. import db
from app.models import ExternalBlogAccount

# ---------------- 共通定数 ----------------
LANDING_URL   = "https://note.com/signup?signup_type=email"
FORM_PATH     = "/signup/form"
COMPLETE_PATH = "/signup/complete"
STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# =============================================================================
# 0. ユーティリティ
# =============================================================================
def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _wait(a: float = 0.6, b: float = 1.3) -> None:
    """軽いランダムウェイト ― bot 検知回避用"""
    time.sleep(random.uniform(a, b))


# =============================================================================
# 本番用：ExternalBlogAccount を渡してサインアップ
# =============================================================================
async def signup_note_account(account: ExternalBlogAccount) -> Dict[str, str | bool]:
    """
    Parameters
    ----------
    account : ExternalBlogAccount
        登録済み email / password / nickname を保持しているモデル

    Returns
    -------
    dict
        {"ok": True} もしくは {"ok": False, "error": "..."}
    """
    if not account.email or not account.password:
        return {"ok": False, "error": "email/password not set"}

    nickname = account.nickname or f"user-{_rand(6)}"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                ],
                slow_mo=200,  # bot 検知を避けるため少し待つ
            )
            ctx = await browser.new_context(
                locale="ja-JP",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            )
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # 1) メール登録ページ → フォーム
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle")
            try:
                await page.locator("text=メールで登録").first.click(timeout=5_000)
            except PWTimeout:
                pass
            await page.wait_for_url(f"**{FORM_PATH}**", timeout=15_000)

            # 2) フォーム入力
            await page.fill('input[type="email"]', account.email)
            await page.fill('input[type="password"]', account.password)
            if await page.locator('input[name="nickname"]').count():
                await page.fill('input[name="nickname"]', nickname)
            cb = page.locator('input[type="checkbox"]')
            if await cb.count():
                await cb.check()

            submit = page.locator('[data-testid="signup-submit"]')
            # ── Playwright v1.43+：arg= で渡す ──
            await page.wait_for_function("el => !el.disabled", arg=submit, timeout=30_000)
            await submit.click()
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)

            # 3) storage_state 保存
            state_dir = Path("storage_states")
            state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(
                json.dumps(await ctx.storage_state(), ensure_ascii=False)
            )

            # 4) DB 更新
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()

        logging.info("[note_signup] SUCCESS: %s", account.email)
        return {"ok": True}

    except Exception as e:  # noqa: BLE001
        logging.error("[note_signup] FAILED: %s", e)
        return {"ok": False, "error": str(e)}


__all__ = ["signup_note_account"]
