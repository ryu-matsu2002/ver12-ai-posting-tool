# -*- coding: utf-8 -*-
"""
Note.com アカウント自動登録モジュール（本番用・2025-07-07 改訂）
====================================================================
signup_note_account(account: ExternalBlogAccount) -> {"ok":True} |
                                             {"ok":False, "error":...}

変更点
--------------------------------------------------------------------
1.  フォーム検出を「iframe 全走査 → メイン DOM 再走査」の二段階に。
2.  ボタン活性化待機を wait_for_selector('[data-testid=signup-submit]:not([disabled])')
3.  失敗時にスクリーンショットを保存（storage_states/signup_fail_<id>.png）
4.  ステップ毎の詳細ログを追加。
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
from .mail_tm_client import create_inbox, wait_verify_link

# --------------------------------------------------------------------
# 固定定数
# --------------------------------------------------------------------
LANDING_URL   = "https://note.com/signup?signup_type=email"
FORM_PATH     = "/signup/form"
COMPLETE_PATH = "/signup/complete"

STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.navigator.chrome={runtime:{}};
Object.defineProperty(navigator,'languages',{get:()=>['ja-JP','ja']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
"""

# --------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------
def _rand(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _wait(a: float = .6, b: float = 1.3) -> None:
    time.sleep(random.uniform(a, b))

# --------------------------------------------------------------------
# メイン関数
# --------------------------------------------------------------------
async def signup_note_account(account: ExternalBlogAccount) -> Dict[str, str | bool]:
    """
    Parameters
    ----------
    account : ExternalBlogAccount
        email / password / nickname が設定済みのモデル

    Returns
    -------
    dict
        {"ok": True} もしくは {"ok": False, "error": "..."}
    """
    # ① ここでワンタイムメール発行し、モデルにセット --------------★ NEW
    email, mail_pwd, token = create_inbox()
    account.email = email
    account.password = _rand(12) + "A1!"
    nickname = account.nickname or f"user-{_rand(6)}"
    db.session.commit()   # email/password を DB に確定

    
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
                slow_mo=200,   # bot 検知回避：少しウェイト
            )
            ctx = await browser.new_context(locale="ja-JP")
            page = await ctx.new_page()
            await page.add_init_script(STEALTH_JS)

            # ────────────────────────────────────────────────
            # 1) LP → ページロード
            # ────────────────────────────────────────────────
            await page.goto(LANDING_URL, timeout=60_000)
            await page.wait_for_load_state("domcontentloaded")
            logging.info("[note_signup] step=lp-loaded id=%s", account.id)

            # ────────────────────────────────────────────────
            # 2) フォーム検出（iframe 全走査 → DOM）
            # ────────────────────────────────────────────────
            logging.info("[note_signup] step=form-scan id=%s", account.id)

            email_sel = "input[type='email']"
            pwd_sel   = "input[type='password']"

            target_email = None
            target_pwd   = None

            # a) iframe をすべて走査
            for f in page.frames:
                try:
                    if await f.locator(email_sel).count() and await f.locator(pwd_sel).count():
                        target_email = f.locator(email_sel).first
                        target_pwd   = f.locator(pwd_sel).first
                        logging.info("[note_signup] ✓ form in iframe src=%s", f.url)
                        break
                except Exception:
                    continue   # cross-origin 等でアクセス出来ない iframe は無視

            # b) 見つからなければメイン DOM をチェック
            if not target_email:
                # 「メールで登録」ボタンがあれば押下
                if await page.locator("text=メールで登録").count():
                    await page.locator("text=メールで登録").first.click()
                    await page.wait_for_timeout(1000)

                if await page.locator(email_sel).count() and await page.locator(pwd_sel).count():
                    target_email = page.locator(email_sel).first
                    target_pwd   = page.locator(pwd_sel).first
                    logging.info("[note_signup] ✓ form in main DOM id=%s", account.id)

            # c) それでも無ければスクショを撮ってエラー
            # ③ 入力フォームが見つかった後
            if not target_email:
                fail_png = Path("storage_states") / f"signup_fail_{account.id}.png"
                fail_png.parent.mkdir(exist_ok=True)
                await page.screenshot(path=str(fail_png))
                raise RuntimeError("signup form not found")

            # ✅ フォームの存在する frame コンテキストを取得
            form_ctx = target_email.page  # これは target_email が属する frame/Page を返す

            # ③ 入力 → 送信
            await target_email.fill(account.email)
            await target_pwd.fill(account.password)

            # ✅ フォーム内で submit ボタンを探す（frame 上）
            submit_btn = form_ctx.locator(
                "[data-testid='signup-submit']:not([disabled]), "
                "button[type='submit']:not([disabled]), "
                "button:has-text('登録'), "
                "button:has-text('サインアップ'), "           # ★追加
                "button:has-text('メールアドレスで登録')"      # ★追加
            ).first
            await submit_btn.wait_for(state="visible", timeout=60_000)
            await submit_btn.click()

            await page.wait_for_url(f"**{COMPLETE_PATH}**", timeout=60_000)
            logging.info("[note_signup] step=complete-page id=%s", account.id)

            # ② メール認証リンクを待ち受けて開く ----------------------★ NEW
            verify_link = wait_verify_link(token, timeout=180)
            if not verify_link:
                raise RuntimeError("verification mail timeout")
            await page.goto(verify_link, timeout=60_000)
            await page.wait_for_selector("text=メール認証が完了", timeout=30_000)


            # ────────────────────────────────────────────────
            # 4) storage_state 保存
            # ────────────────────────────────────────────────
            state_dir  = Path("storage_states"); state_dir.mkdir(exist_ok=True)
            state_path = state_dir / f"note_{account.id}.json"
            state_path.write_text(json.dumps(await ctx.storage_state(), ensure_ascii=False))

            # ────────────────────────────────────────────────
            # 5) DB 更新
            # ────────────────────────────────────────────────
            account.nickname    = nickname
            account.cookie_path = str(state_path)
            account.status      = "active"
            account.created_at  = datetime.utcnow()
            db.session.commit()

            await browser.close()

        logging.info("[note_signup] ✅ SUCCESS id=%s", account.id)
        return {"ok": True}

    # ── ファイル末尾付近 ─────────────────────────────────────────────
    except (PWTimeout, Exception) as e:          # noqa: BLE001
        logging.error("[note_signup] ❌ FAILED id=%s %s", account.id, e)

        # ★ 失敗時に必ず PNG を残す
        try:
            err_png = Path("storage_states") / f"signup_fail_{account.id}.png"
            err_png.parent.mkdir(exist_ok=True)
            await page.screenshot(path=str(err_png))
            logging.info("[note_signup] 📸 saved => %s", err_png)
        except Exception:
            logging.warning("[note_signup] screenshot failed")

        _mark_error(account, str(e))
        return {"ok": False, "error": str(e)}

    
# ---------------------------------------------------------------
# 共通エラーハンドラ  ★NEW
def _mark_error(acct: ExternalBlogAccount, msg: str):
    acct.status  = "error"
    acct.message = msg[:255]
    db.session.commit()

# 外部公開
__all__ = ["signup_note_account"]
