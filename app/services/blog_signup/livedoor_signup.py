"""
ライブドアブログ アカウント自動登録
==================================
* Playwright でフォーム入力 → 仮登録メール受信 → 本登録 → API キー取得
* 使い捨てメール: mail.gw（create_inbox で発行）
* 取得した API Key／Blog ID を ExternalBlogAccount に保存
*
* 2025-07-09 改訂:
*  - 送信直後に URL／タイトル／成功メッセージを検証
*  - CAPTCHA iframe 検出ログを追加
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
from playwright.async_api import (Page, TimeoutError as PwTimeout,
                                  async_playwright)

from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.blog_signup.crypto_utils import encrypt
from app.services.livedoor.llm_helper import extract_form_fields
from app.services.mail_utils.mail_gw import (create_inbox, poll_latest_link)

logger = logging.getLogger(__name__)

SIGNUP_URL = "https://member.livedoor.com/register/input"
SUCCESS_PATTERNS: List[str] = ["メールを送信しました", "仮登録"]  # 送信完了画面に含まれる文言


# ──────────────────────────────────────────────────────────────
async def _fill_form_with_llm(page: Page, hints: Dict[str, str]) -> None:
    """
    livedoor 登録画面の各入力欄に値を埋め込む。

    GPT-4o によるフィールド推定結果（label → selector）の
    マッピング `extract_form_fields()` を流用。
    """
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


# ──────────────────────────────────────────────────────────────
async def _signup_internal(
    email: str,
    token: str,
    password: str,
    nickname: str,
) -> Dict[str, str]:
    """
    実際のブラウザ操作を行うコルーチン。

    Returns
    -------
    dict
        {"blog_id": <str>, "api_key": <str>}
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        page = await browser.new_page()

        # 1) 会員登録フォームへ遷移
        await page.goto(SIGNUP_URL, timeout=30_000)

        # CAPTCHA iframe が最初から出ていないか確認
        captcha_present = await page.query_selector("iframe[src*='recaptcha']") is not None
        logger.info("[LD-Signup] captcha_present=%s", captcha_present)

        await _fill_form_with_llm(
            page,
            {
                "メールアドレス": email,
                "パスワード": password,
                "パスワード(確認)": password,
                "ニックネーム": nickname,
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
            html_path = Path("/tmp/ld_signup_debug.html")
            png_path = Path("/tmp/ld_signup_debug.png")
            html_path.write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path=str(png_path), full_page=True)
            await browser.close()
            raise RuntimeError(
                f"登録フォームの送信ボタンが見つからず失敗。HTML: {html_path}, PNG: {png_path}"
            )

        # クリック後にネットワークが静まるまで待ち、現在の URL／タイトルをログ
        await page.wait_for_load_state("networkidle")
        title_now = await page.title()
        logger.info("[LD-Signup] after submit url=%s title=%s", page.url, title_now)

        # 送信成功判定
        html_after = await page.content()
        if not any(pat in html_after for pat in SUCCESS_PATTERNS):
            bad_html = Path("/tmp/ld_signup_post_submit.html")
            bad_png = Path("/tmp/ld_signup_post_submit.png")
            bad_html.write_text(html_after, encoding="utf-8")
            await page.screenshot(path=str(bad_png), full_page=True)
            await browser.close()
            raise RuntimeError(
                f"送信後に成功メッセージが見当たらない。HTML: {bad_html}, PNG: {bad_png}"
            )
        # -----------------------------------------------------------

        # 2) メール認証リンクを取得
        link = poll_latest_link(
            token,
            pattern=r"https://member\.livedoor\.com/register/.*",
            timeout=180,
        )
        if not link:
            await browser.close()
            raise RuntimeError("メール認証リンクを取得できませんでした")

        logger.info("[LD-Signup] verification link=%s", link)
        await page.goto(link, timeout=30_000)

        # 3) ブログ開設（自動リダイレクトを待つ）
        await page.wait_for_url(re.compile(r"https://blog\.livedoor\.com/.*"), timeout=60_000)

        # 4) blog_id を抽出
        m = re.search(r"https://(.+?)\.blogcms\.jp", page.url)
        if not m:
            await browser.close()
            raise RuntimeError("blog_id を URL から抽出できませんでした")
        blog_id = m.group(1)

        # 5) API Key を生成／取得
        await page.goto("https://blog.livedoor.com/settings/api", timeout=30_000)
        if await page.is_visible("text=APIキーを生成"):
            await page.click("text=APIキーを生成", timeout=15_000)
            await page.wait_for_selector("input[name='apikey']", timeout=15_000)

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
    email, token = create_inbox()
    logger.info("[LD-Signup] disposable email = %s", email)
    logger.info("[LD-Signup] mailgw jwt = %s", token)

    password = "Ld" + str(int(time.time()))
    nickname = site.name[:10]

    try:
        res = asyncio.run(_signup_internal(email, token, password, nickname))
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


# --- 互換エイリアス（tasks.py から import される） -------------------
def signup(site, email_seed: str = "ld") -> ExternalBlogAccount:
    """tasks.py 用ラッパー"""
    return register_blog_account(site, email_seed=email_seed)
