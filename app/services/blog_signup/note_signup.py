# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール

・legacy_signup_note() …… mail.tm を使った検証用（従来）
・signup_note_account(account) …… ExternalBlogAccount を受け取り、
  Playwright だけでサインアップし storage_state.json を保存する【本番用】
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

# ── Playwright ────────────────────────────────────────────────────────────
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as SyncPWTimeout,
    Error as SyncPWError,
    Page,
)

# ── プロジェクト内部 ──────────────────────────────────────────────────────
from .. import db  # blog_signup パッケージの __init__.py から再エクスポート済み想定
from app.models import ExternalBlogAccount

# ---------------- 共通定数 ----------------
LANDING_URL = "https://note.com/signup?signup_type=email"
FORM_PATH = "/signup/form"
COMPLETE_PATH = "/signup/complete"
STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""


# =============================================================================
# 1) 既存の mail.tm 利用スクリプトを legacy_* として残す
# =============================================================================
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link

__all__ = [
    "signup_note_account",  # 本番用
    "legacy_signup_note",   # 検証用
]


def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _wait(a: float = 0.6, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))


def legacy_signup_note() -> Dict[str, str | bool | None]:
    """
    mail.tm を使って完全に新規の Note アカウントを作成する旧関数。
    開発検証以外では使用しない。
    戻り値: {"ok": bool, "email": str|None, "password": str|None, "error": str|None}
    """
    email, jwt = create_inbox()
    password = _rand(12) + "!"
    nickname = f"user-{_rand(6)}"

    logging.info("[note_signup] new inbox %s", email)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                    "--disable-extensions",
                ],
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
            page.add_init_script(STEALTH_JS)

            # ── フォーム入力 ──
            page.goto(LANDING_URL, timeout=30_000)
            page.wait_for_load_state("networkidle")
            _wait()
            try:
                page.locator("text=メールで登録").first.click(timeout=5_000)
            except SyncPWTimeout:
                pass
            page.wait_for_url(f"**{FORM_PATH}**", timeout=15_000)

            page.fill('input[type="email"]', email)
            _wait()
            page.fill('input[type="password"]', password)
            _wait()
            if page.locator('input[name="nickname"]').count():
                page.fill('input[name="nickname"]', nickname)
            cb = page.locator('input[type="checkbox"]')
            if cb.count():
                cb.check()

            submit_btn = page.locator('button[type="submit"]')
            for _ in range(60):
                if submit_btn.is_enabled():
                    break
                time.sleep(0.5)
            submit_btn.click()
            page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)
            browser.close()

        verify_link = poll_latest_link(jwt, sender_like="@note.com")
        if not verify_link:
            return {"ok": False, "email": email, "password": password, "error": "verification mail not found"}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(verify_link, timeout=30_000)
            page.wait_for_load_state("networkidle")
            browser.close()

        logging.info("[note_signup] SUCCESS: %s", email)
        return {"ok": True, "email": email, "password": password, "error": None}

    except (SyncPWTimeout, SyncPWError) as e:
        logging.error("[note_signup] Playwright error: %s", e)
        return {"ok": False, "email": email, "password": password, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        logging.exception("[note_signup] Unexpected error")
        return {"ok": False, "email": email, "password": password, "error": str(e)}


# =============================================================================
# 2) 本番用: ExternalBlogAccount を登録し storage_state を保存
# =============================================================================
async def signup_note_account(account: ExternalBlogAccount) -> None:
    """
    ExternalBlogAccount を受け取り、
    その email / password / nickname で Note にサインアップして
    storage_state.json を保存し、account.cookie_path & nickname を更新する。
    """
    if not account.email or not account.password:
        raise ValueError("ExternalBlogAccount に email / password が設定されていません")

    nickname = (
        account.nickname
        if account.nickname
        else f"user-{_rand(6)}"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-gpu",
            ],
            slow_mo=200,  # スローモーションで bot 検知を回避
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

        # ── 1. メール登録画面 → フォーム ─────────────────────
        await page.goto(LANDING_URL, timeout=30_000)
        await page.wait_for_load_state("networkidle")
        try:
            await page.locator("text=メールで登録").first.click(timeout=5_000)
        except PWTimeout:
            pass
        await page.wait_for_url(f"**{FORM_PATH}**", timeout=15_000)

        # ── 2. フォーム入力 ────────────────────────────────
        await page.fill('input[type="email"]', account.email)
        await page.fill('input[type="password"]', account.password)
        if await page.locator('input[name="nickname"]').count():
            await page.fill('input[name="nickname"]', nickname)
        cb = page.locator('input[type="checkbox"]')
        if await cb.count():
            await cb.check()

        submit = page.locator('button[type="submit"]')
        for _ in range(60):
            if await submit.is_enabled():
                break
            await asyncio.sleep(0.5)
        await submit.click()
        await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)

        # ── 3. storage_state.json 保存 ───────────────────────
        storage_state = await ctx.storage_state()
        Path("storage_states").mkdir(exist_ok=True)
        state_path = Path("storage_states") / f"note_{account.id}.json"
        state_path.write_text(json.dumps(storage_state, ensure_ascii=False))

        # ── 4. DB 更新 ──────────────────────────────────────
        account.nickname = nickname
        account.cookie_path = str(state_path)
        account.status = "active"
        account.created_at = datetime.utcnow()
        db.session.commit()

        await browser.close()
