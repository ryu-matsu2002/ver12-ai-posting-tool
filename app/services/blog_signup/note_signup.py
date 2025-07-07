# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（堅牢版）
=============================================================
signup_note_account(account) → {"ok": True} | {"ok": False, "error": …}

✅ 改修ポイント
----------------------------------------------------------------
1. 「メールで登録」ボタンを押したあと **実際に e-mail フォームを含む
   iframe を動的に特定** する方式に変更。
2. `data-testid="signup-submit"` が enabled になるまで待機。
3. 各ステップで詳細ログを出力。
"""

from __future__ import annotations
import json, logging, random, string, time
from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from .. import db
from app.models import ExternalBlogAccount

# ---------------------------------------------------------------------------
LANDING_URL     = "https://note.com/signup?signup_type=email"
COMPLETE_PATH   = "/signup/complete"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# ---------------------------------------------------------------------------

def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float = .6, b: float = 1.2) -> None:
    time.sleep(random.uniform(a, b))

# ---------------------------------------------------------------------------

async def signup_note_account(account: ExternalBlogAccount) -> Dict[str, str | bool]:
    """Playwright を使って Note のサインアップを完了し storage_state を保存"""
    if not (account.email and account.password):
        return {"ok": False, "error": "email/password not set"}

    nickname = account.nickname or f"user-{_rand(6)}"
    logging.info("[note_signup] ▶ START id=%s", account.id)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
                slow_mo=150,         # ゆっくり操作して bot 検知を回避
            )
            ctx   = await browser.new_context(locale="ja-JP")
            page  = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # ── 1) サインアップランディング ─────────────────────────
            await page.goto(LANDING_URL, timeout=30_000)
            await page.wait_for_load_state("domcontentloaded")

            # フォームが最初からあるか？
            email_input = page.query_selector("input[type='email']")
            if not email_input:
                # 「メールで登録」ボタンをクリック
                btn = page.locator("text=メールで登録").first
                await btn.click()
                _wait()

            # ── 2) e-mail / password フォームを含む iframe を動的に決定 ──
            target_frame = None
            for f in page.frames:
                if await f.query_selector("input[type='email']"):
                    target_frame = f
                    break

            if not target_frame:
                raise RuntimeError("e-mail form iframe not found")

            logging.info("[note_signup] iframe found → %s", target_frame.name)

            # ── 3) 入力 & 送信 ─────────────────────────────
            await target_frame.fill("input[type='email']", account.email, timeout=10_000)
            await target_frame.fill("input[type='password']", account.password)

            nick = await target_frame.query_selector("input[name='nickname']")
            if nick:
                await nick.fill(nickname)

            # submit ボタンが enabled になるまで待つ
            await target_frame.wait_for_selector(
                "[data-testid='signup-submit']:not([disabled])", timeout=30_000
            )
            await target_frame.click("[data-testid='signup-submit']")

            # 完了ページ遷移を確認
            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=30_000)
            logging.info("[note_signup] form submitted OK")

            # ── 4) storage_state 保存 ─────────────────────
            state_dir  = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # ── 5) DB 更新 ────────────────────────────────
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()
            logging.info("[note_signup] ✅ SUCCESS id=%s", account.id)
            return {"ok": True}

    except PWTimeout as e:
        msg = f"Timeout: {e}"
    except Exception as e:          # noqa: BLE001
        msg = str(e)

    # 失敗時
    logging.error("[note_signup] ❌ FAILED id=%s %s", account.id, msg)
    return {"ok": False, "error": msg}

__all__ = ["signup_note_account"]
