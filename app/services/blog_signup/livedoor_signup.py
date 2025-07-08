"""
ライブドアブログ アカウント自動登録
==================================
* Playwright + GPT でフォーム入力
* メールは mail.tm → 1secmail に切替
* 取得した API Key を ExternalBlogAccount に保存
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Dict

from playwright.async_api import async_playwright, Page
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.livedoor.llm_helper import extract_form_fields
from app.services.blog_signup.crypto_utils import encrypt
from app.services.blog_signup.mail_tm_client import (
     create_disposable_email,
     poll_inbox,
)

logger = logging.getLogger(__name__)

SIGNUP_URL = "https://member.livedoor.com/register/input"

# ──────────────────────────────────────────────────────────────
async def _fill_form_with_llm(page: Page, hints: Dict[str, str]) -> None:
    html = await page.content()
    mapping = extract_form_fields(html)
    for field in mapping:
        sel = field["selector"]
        label = field["label"]
        value = hints.get(label, "")
        if not value:
            continue
        try:
            await page.fill(sel, value)
        except Exception:
            logger.warning("failed to fill %s (%s)", label, sel)


# livedoor_signup.py  ── 修正後の _signup_internal() 〈省略なし〉
# ※ 冒頭 import に以下が追加されていることを確認してください
from pathlib import Path
from playwright.async_api import TimeoutError as PwTimeout

async def _signup_internal(
    email: str,
    token: str,
    password: str,
    nickname: str,
) -> Dict[str, str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--no-sandbox"])
        page    = await browser.new_page()

        # 1)  会員登録フォームへ遷移
        await page.goto(SIGNUP_URL, timeout=30_000)

        # GPT で推定したセレクタを使って自動入力
        await _fill_form_with_llm(
            page,
            {
                "メールアドレス":    email,
                "パスワード":        password,
                "パスワード(確認)":   password,
                "ニックネーム":      nickname,
            },
        )

        # ---- 送信ボタンを押す（複数セレクタを順に試す）-------------
        await page.wait_for_load_state("networkidle")

        selectors = [
            "input[type='submit'][value*='ユーザー情報を登録']",
            "input[type='submit']",
            "button:has-text('確認メール')",
            "button.c-btn--primary",
        ]

        clicked = False
        for sel in selectors:
            if await page.is_visible(sel):
                try:
                    await page.click(sel, timeout=5_000)
                    clicked = True
                    break
                except PwTimeout:
                    pass  # selector は見えるがクリック不可 → 次へ

        if not clicked:
            # デバッグ用に HTML とスクショを保存して例外送出
            html_path = Path("/tmp/ld_signup_debug.html")
            html_path.write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path="/tmp/ld_signup_debug.png", full_page=True)
            await browser.close()
            raise RuntimeError(
                f"登録フォームの送信ボタンが見つからず失敗。HTML 保存: {html_path}"
            )
        # -----------------------------------------------------------

        # 2)  メール認証リンクを取得
        link = poll_inbox(
            token,
            pattern=r"https://member\.livedoor\.com/register/.*",
        )
        await page.goto(link, timeout=30_000)

        # 3)  ブログ開設（自動リダイレクトを待つ）
        await page.wait_for_url(re.compile(r"https://blog\.livedoor\.com/.*"))

        # 4)  blog_id を抽出
        m        = re.search(r"https://(.+?)\.blogcms\.jp", page.url)
        blog_id  = m.group(1)

        # 5)  API Key を生成
        await page.goto("https://blog.livedoor.com/settings/api", timeout=30_000)
        if await page.is_visible("text=APIキーを生成"):
            await page.click("text=APIキーを生成")
            await page.wait_for_selector("input[name='apikey']")

        api_key = await page.input_value("input[name='apikey']")

        await browser.close()
        return {"blog_id": blog_id, "api_key": api_key}



# ──────────────────────────────────────────────────────────────
def register_blog_account(site, email_seed: str = "ld") -> ExternalBlogAccount:
    """
    外部呼び出し関数
    -------------
    * Site オブジェクトを受け取り、ExternalBlogAccount を新規作成
    * 既にアカウントがある場合はそのまま返す
    """
    account = (
        ExternalBlogAccount.query.filter_by(
            site_id=site.id, blog_type=BlogType.LIVEDOOR
        ).first()
    )
    if account:
        return account

    # 使い捨てメールを発行
    email, token = create_disposable_email(seed=email_seed)
    password = "Ld" + str(int(time.time()))  # シンプルでOK
    nickname = site.name[:10]

    try:
        # ThreadPoolExecutor 内でも安全に実行できるワンショット実行
        res = asyncio.run(
            _signup_internal(email, token, password, nickname)
        )
    except Exception as e:
        logger.exception("[LD-Signup] failed: %s", e)
        raise


    # DB 保存
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
# --- ここから追加 ----------------------------------------------------
def signup(site, email_seed: str = "ld"):
    """
    互換ラッパー（tasks.py が import するため）
    """
    return register_blog_account(site, email_seed=email_seed)
# --- 追加ここまで ----------------------------------------------------
