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


async def _signup_internal(email: str, token: str, password: str, nickname: str) -> Dict[str, str]:
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await br.new_page()

        # 1) 会員登録フォーム
        await page.goto(SIGNUP_URL, timeout=30000)
        await _fill_form_with_llm(
            page,
            {
                "メールアドレス": email,
                "パスワード": password,
                "パスワード(確認)": password,
                "ニックネーム": nickname,
            },
        )
        await page.click("button[type='submit']")

        # 2) メール認証リンクを取得
        link = poll_inbox(token, pattern=r"https://member\.livedoor\.com/register/.*")
        await page.goto(link, timeout=30000)

        # 3) ブログ開設（自動リダイレクトで完了）
        await page.wait_for_url(re.compile(r"https://blog\.livedoor\.com/.*"))

        # 4) blog_id を抽出
        m = re.search(r"https://(.+?)\.blogcms\.jp", page.url)
        blog_id = m.group(1)

        # 5) API Key を生成
        await page.goto("https://blog.livedoor.com/settings/api", timeout=30000)
        # 「APIキーを生成」ボタンがあれば押す
        if await page.is_visible("text=APIキーを生成"):
            await page.click("text=APIキーを生成")
            await page.wait_for_selector("input[name='apikey']")

        api_key = await page.input_value("input[name='apikey']")
        await br.close()
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
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(_signup_internal(email, token, password, nickname))
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
