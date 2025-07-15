"""
ライブドアブログ アカウント自動登録
==================================
* Playwright でフォーム入力 → 仮登録メール受信 → 本登録 → API キー取得
* 使い捨てメール: mail.gw（create_inbox で発行）
* 取得した API Key／Blog ID を ExternalBlogAccount に保存
*
* 2025-07-09 改訂:
*  - CAPTCHA 画像を自前OCRで自動入力（captcha_solver.solve）
*  - 送信直後に URL／タイトル／成功メッセージを検証
*  - クリック出来ない場合に HTML／PNG を /tmp に保存
*  - 詳細ログを強化しデバッグ容易化
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from playwright.async_api import async_playwright, Page
from playwright.async_api import TimeoutError as PwTimeout

from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.blog_signup.crypto_utils import encrypt
from app.services.livedoor.llm_helper import extract_form_fields
from app.services.mail_utils.mail_gw import create_inbox, poll_latest_link_gw
from app.services.captcha_solver import solve  # ←★ 追加
# imports（上部）

logger = logging.getLogger(__name__)

SIGNUP_URL = "https://member.livedoor.com/register/input"
SUCCESS_PATTERNS: List[str] = ["メールを送信しました", "仮登録"]

# ──────────────────────────────────────────────────────────────
async def _fill_form_with_llm(page: Page, hints: Dict[str, str]) -> None:
    """GPT で推定したセレクタに値を流し込む"""
    html = await page.content()
    mapping = extract_form_fields(html)
    logger.info("🧠 フィールド推論結果: %s", mapping)
    for field in mapping:
        sel = field["selector"]
        label = field["label"]
        value = hints.get(label, "")
        if not value:
            logger.info("⚠️ 値が未設定のフィールド: %s", label)
            continue
        try:
            await page.fill(sel, value)
            logger.info("✅ フィールド '%s' に値 '%s' を入力しました", sel, value)
        except Exception as e:
            logger.warning("⚠️ fill失敗 label='%s' selector='%s' → %s", label, sel, str(e))


# ──────────────────────────────────────────────────────────────
async def _signup_internal(
    email: str,
    token: str,
    password: str,
    nickname: str,
) -> Dict[str, str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page()

        # 1) フォームへ遷移
        await page.goto(SIGNUP_URL, timeout=30_000)

        # 🔧 フィールドに手動で入力（推論は使わない）
        await page.fill("input[name='email']", email)
        await page.fill("input[name='password']", password)
        await page.fill("input[name='password2']", password)
        await page.fill("input[name='nickname']", nickname)
        logger.info("✅ 手動で全フィールドに入力完了（email=%s, nickname=%s）", email, nickname)


        # 画像CAPTCHAがある場合は自動で解く
        if await page.is_visible("img[src*='captcha']"):
            for attempt in range(3):  # 最大3回
                img_bytes = await page.locator("img[src*='captcha']").screenshot()
                text = solve(img_bytes)
                await page.fill("input[name='captcha']", text)
                logger.info("[LD-Signup] solve captcha try%d='%s'", attempt + 1, text)

                # 送信ボタン押下
                if await page.is_visible("input[type='submit']"):
                    await page.click("input[type='submit']")
                else:
                    await page.click("button.c-btn--primary")

                # 成功判定：エラーメッセージが空
                await page.wait_for_load_state("networkidle")
                if not await page.is_visible("#captcha_msg:not(:empty)"):
                    break   # 成功
                # 失敗 → 画像をクリックしてリフレッシュして再挑戦
                await page.click("img[src*='captcha']")

        # ---- CAPTCHA が無い or 入力済み状態で送信ボタン確実クリック ----
        await page.wait_for_load_state("networkidle")
        clicked = False
        for sel in [
            "input[type='submit'][value*='ユーザー情報を登録']",
            "input[type='submit']",
            "button:has-text('確認メール')",
            "button.c-btn--primary",
        ]:
            if await page.is_visible(sel):
                try:
                    await page.click(sel, timeout=5_000)
                    clicked = True
                    break
                except PwTimeout:
                    pass
        if not clicked:
            html = Path("/tmp/ld_signup_debug.html")
            png  = Path("/tmp/ld_signup_debug.png")
            html.write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path=str(png), full_page=True)
            await browser.close()
            raise RuntimeError(f"送信ボタンが押せず失敗。HTML:{html} PNG:{png}")

        await page.wait_for_load_state("networkidle")
        logger.info("[LD-Signup] after submit url=%s title=%s", page.url, await page.title())

        # 成功文言チェック
        content = await page.content()
        if not any(pat in content for pat in SUCCESS_PATTERNS):
            bad = Path("/tmp/ld_signup_post_submit.html")
            bad.write_text(await page.content(), encoding="utf-8")
            await browser.close()
            raise RuntimeError(f"送信後に成功メッセージが無い → {bad}")

        # 2) 認証リンク
        link = None

        # ✅ poll_latest_link_gw の正体を確認するログ
        logger.info("✅ poll_latest_link_gw の参照先: %s", poll_latest_link_gw)
        logger.info("✅ poll_latest_link_gw の型: %s", type(poll_latest_link_gw))

        # 🔽🔽 この行を追加してください 🔽🔽
        logger.info("✅ 使用中の poll_latest_link_gw = %s", poll_latest_link_gw)
        import inspect

        logger.info("💡 poll_latest_link_gw type: %s", type(poll_latest_link_gw))
        logger.info("💡 poll_latest_link_gw is async generator: %s", inspect.isasyncgenfunction(poll_latest_link_gw))
        logger.info("💡 poll_latest_link_gw() is async generator object: %s", inspect.isasyncgen(poll_latest_link_gw(token)))

        async for l in poll_latest_link_gw(token, r"https://member\.livedoor\.com/register/.*", 180):
            link = l
            break


        if not link:
            await browser.close()
            raise RuntimeError("メール認証リンクが取得できません")
        logger.info("[LD-Signup] verification link=%s", link)
        await page.goto(link, timeout=30_000)

        # 3) 自動リダイレクトを待つ
        import re as regex  # ← 別名で re を再定義してみてもよい

        pattern = regex.compile(r"https://blog\.livedoor\.com/.*")
        await page.wait_for_url(lambda url: bool(pattern.match(url)), timeout=60_000)


        # 4) blog_id
        m = re.search(r"https://(.+?)\.blogcms\.jp", page.url)
        if not m:
            await browser.close()
            raise RuntimeError("blog_id が抽出できませんでした")
        blog_id = m.group(1)

        # 5) APIキー取得
        await page.goto("https://blog.livedoor.com/settings/api", timeout=30_000)
        if await page.is_visible("text=APIキーを生成"):
            await page.click("text=APIキーを生成")
            await page.wait_for_selector("input[name='apikey']")
        api_key = await page.input_value("input[name='apikey']")

        await browser.close()
        return {"blog_id": blog_id, "api_key": api_key}

# ──────────────────────────────────────────────────────────────
def register_blog_account(site, email_seed: str = "ld") -> ExternalBlogAccount:
    import nest_asyncio
    nest_asyncio.apply()  # ✅ イベントループ重複実行を許可（必須）

    account = ExternalBlogAccount.query.filter_by(
        site_id=site.id, blog_type=BlogType.LIVEDOOR
    ).first()
    if account:
        return account

    email, token = create_inbox()
    logger.info("[LD-Signup] disposable email = %s", email)

    password = "Ld" + str(int(time.time()))
    nickname = site.name[:10]

    try:
        # ✅ asyncio.run() は1回のみ、nest_asyncioにより安全に呼び出し
        res = asyncio.run(_signup_internal(email, token, password, nickname))

    except Exception as e:
        logger.error("[LD-Signup] failed: %s", str(e))
        raise

    new_account = ExternalBlogAccount(
        site_id=site.id,
        blog_type=BlogType.LIVEDOOR,
        email=email,
        username=nickname,
        password=password,
        livedoor_blog_id=res["blog_id"],
        atompub_key_enc=encrypt(res["api_key"]),
        api_post_enabled=True,
        nickname=nickname,
    )
    db.session.add(new_account)
    db.session.commit()
    return new_account

# 互換ラッパー
def signup(site, email_seed: str = "ld"):
    return register_blog_account(site, email_seed=email_seed)
