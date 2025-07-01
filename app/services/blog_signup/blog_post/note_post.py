# app/services/blog_post/note_poster.py
"""
Note へ記事を投稿する簡易ユーティリティ
  - Playwright でエディタに貼り付け → 公開ボタン
  - 投稿 URL を返す
"""
import asyncio, logging, re
from datetime import datetime
from playwright.async_api import async_playwright

async def _post(title: str, body_html: str, email: str, password: str) -> str:
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        ctx = await br.new_context()
        pg  = await ctx.new_page()

        # --- 1. ログイン ---
        await pg.goto("https://note.com/login")
        await pg.fill('input[type="email"]', email)
        await pg.click('button[type="submit"]')
        await pg.fill('input[name="password"]', password)
        await pg.click('button[type="submit"]')
        await pg.wait_for_load_state("networkidle")

        # --- 2. 新規投稿ページへ ---
        await pg.goto("https://note.com/new")
        await pg.wait_for_selector('[placeholder="タイトル"]')
        await pg.fill('[placeholder="タイトル"]', title)

        # --- 3. 本文 iframe 内に HTML を直接書き込む ---
        iframe = pg.frame_locator('iframe').first
        await iframe.locator('div').click()
        # Note のエディタはペーストで HTML を解釈してくれる
        await iframe.locator('div').evaluate("(el, html)=>{el.innerHTML = html}", body_html)

        # --- 4. 公開ボタンをクリック ---
        await pg.click('text=公開')
        await pg.click('text=投稿する')   # モーダルの確認

        # --- 5. 投稿 URL 取得 ---
        await pg.wait_for_url(re.compile(r'https://note.com/.+/n.*'))
        url = pg.url
        await br.close()
        return url

def post_to_note(title: str, body_html: str, email: str, password: str) -> str:
    """同期ラッパー"""
    try:
        return asyncio.run(_post(title, body_html, email, password))
    except Exception as e:
        logging.exception(f"[NotePoster] 投稿失敗: {e}")
        raise
