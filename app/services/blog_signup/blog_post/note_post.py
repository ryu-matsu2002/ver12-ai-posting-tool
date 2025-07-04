# -*- coding: utf-8 -*-
"""
Note へ記事を投稿するユーティリティ（本番版）
=============================================================
post_note_article(account, title, body_html, image_path=None) -> dict
  成功: {"ok": True,  "url": "...", "posted_at": datetime }
  失敗: {"ok": False, "error": "例外文字列"}
-------------------------------------------------------------
* Playwright storage_state.json を利用し、毎回ログインを省略
* 外部SEOジョブから呼ぶことを想定
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from .. import db  # blog_signup パッケージの __init__.py で再エクスポートしている想定
from app.models import ExternalBlogAccount


# ──────────────────────────────────────────
async def _async_post(
    account: ExternalBlogAccount,
    title: str,
    body_html: str,
    image_path: Optional[str] = None,
) -> str:
    """
    Raises
    ------
    Exception : 失敗した場合
    """
    if not account.cookie_path:
        raise ValueError("ExternalBlogAccount.cookie_path が設定されていません")

    cookie_file = Path(account.cookie_path)
    if not cookie_file.exists():
        raise FileNotFoundError(f"cookie_path が見つかりません: {cookie_file}")

    storage_state = json.loads(cookie_file.read_text())

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = await browser.new_context(storage_state=storage_state, locale="ja-JP")
        pg = await ctx.new_page()

        # 1) 新規投稿画面
        await pg.goto("https://note.com/notes/new", timeout=30_000)
        await pg.wait_for_selector('[placeholder="タイトルを入力"]', timeout=15_000)

        # 2) タイトル
        await pg.fill('[placeholder="タイトルを入力"]', title)

        # 3) 本文（contenteditable div）※iframe ではなく直要素に変わっている
        body_area = pg.locator('div[contenteditable="true"]').first
        # フォーカスして HTML を注入
        await body_area.click()
        await body_area.evaluate("(el, html) => { el.innerHTML = html; }", body_html)

        # 4) アイキャッチ画像（任意）
        if image_path:
            file_input = pg.locator('input[type="file"]').first
            await file_input.set_input_files(image_path)

        # 5) 公開
        await pg.click('text=公開設定')      # 右上の公開設定ボタン
        await pg.click('text=公開する')      # モーダル内ボタン
        # 投稿完了 URL 例: https://note.com/<user>/n<hash>
        await pg.wait_for_url(re.compile(r'https://note\.com/.+/n.+'), timeout=30_000)

        url = pg.url
        await browser.close()
        return url


# ──────────────────────────────────────────
def post_note_article(
    account: ExternalBlogAccount,
    title: str,
    body_html: str,
    image_path: Optional[str] = None,
) -> dict:
    """
    外部SEOジョブから呼ばれる同期ラッパー

    Returns
    -------
    dict
        成功: {"ok": True, "url": "...", "posted_at": datetime}
        失敗: {"ok": False, "error": "..."}
    """
    try:
        url = asyncio.run(_async_post(account, title, body_html, image_path))
        # 投稿カウントをインクリメント
        account.posted_cnt += 1
        db.session.commit()
        return {"ok": True, "url": url, "posted_at": datetime.now(timezone.utc)}

    except (PWTimeout, Exception) as e:  # noqa: BLE001
        logging.exception(f"[NotePoster] 投稿失敗: {e}")
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 従来の email+password ログイン版は互換のため残しておく
# （必要なければ削除して OK）
async def _async_post_legacy(title, body_html, email, password) -> str:  # noqa: D401
    from playwright.async_api import async_playwright  # ローカル import
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        pg = await br.new_page()
        await pg.goto("https://note.com/login")
        await pg.fill('input[type="email"]', email)
        await pg.click('button[type="submit"]')
        await pg.fill('input[name="password"]', password)
        await pg.click('button[type="submit"]')
        await pg.wait_for_load_state("networkidle")
        await pg.goto("https://note.com/notes/new")
        await pg.fill('[placeholder="タイトルを入力"]', title)
        body_area = pg.locator('div[contenteditable="true"]').first
        await body_area.evaluate("(el, html)=>{el.innerHTML=html}", body_html)
        await pg.click('text=公開設定')
        await pg.click('text=公開する')
        await pg.wait_for_url(re.compile(r'https://note\.com/.+/n.+'), timeout=30_000)
        url = pg.url
        await br.close()
        return url
