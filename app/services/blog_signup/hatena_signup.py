# -*- coding: utf-8 -*-
"""
Hatena Blog - 新規アカウント自動登録
===============================================
signup_hatena_account(acct) -> {"ok":True}|{"ok":False,"error":...}
* email / password / nickname は ExternalBlogAccount 側で生成済み前提
"""

from __future__ import annotations
import json, logging, random, string, time
from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from .. import db
from app.models import ExternalBlogAccount

# -----------------------------------------------------------------------------
URL_REGISTER = "https://www.hatena.ne.jp/register"
URL_DASH     = "https://blog.hatena.ne.jp/dashboard"      # ログイン後リダイレクト先

def _rand(n:int=6) -> str:
    return "".join(random.choices(string.ascii_lowercase+string.digits,k=n))

# -----------------------------------------------------------------------------
async def signup_hatena_account(acct: ExternalBlogAccount) -> Dict[str,str|bool]:
    log = logging.getLogger("hatena_signup")
    log.info("▶ START id=%s mail=%s", acct.id, acct.email)

    try:
        async with async_playwright() as p:
            br = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = await br.new_context(locale="ja-JP")
            pg  = await ctx.new_page()

            # 1) 登録ページ
            await pg.goto(URL_REGISTER, timeout=30_000)
            await pg.fill("input[name='mail']", acct.email)
            await pg.fill("input[name='password']", acct.password)
            await pg.fill("input[name='password_confirmation']", acct.password)
            await pg.fill("input[name='display_name']", acct.nickname or f"user-{_rand()}")
            # 利用規約チェック
            if await pg.locator("input[name='agreement']").count():
                await pg.check("input[name='agreement']")
            await pg.click("button[type='submit']")
            # → 完全新規の場合メール認証待ちページへ遷移
            await pg.wait_for_load_state("networkidle")

            # 2) そのままダッシュボードへアクセスし cookie 生成
            await pg.goto(URL_DASH, timeout=30_000)

            # 3) storage_state 保存
            state_dir = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"hatena_{acct.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # 4) DB 更新
            acct.cookie_path = str(state_path)
            acct.status      = "active"
            acct.created_at  = datetime.utcnow()
            db.session.commit()

            await br.close()

        log.info("✅ SUCCESS id=%s", acct.id)
        return {"ok": True}

    except (PWTimeout, Exception) as e:          # すべて捕捉して呼び出し側へ返す
        log.error("❌ FAILED id=%s %s", acct.id, e)
        return {"ok": False, "error": str(e)}
