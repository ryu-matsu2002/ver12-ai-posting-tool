# app/services/blog_post/note_poster.py
"""
Note へ記事を投稿するユーティリティ
--------------------------------------------------
post_to_note(title, body_html, email, password) -> dict
  成功: {"ok": True,  "url": "...", "posted_at": datetime }
  失敗: {"ok": False, "error": "例外文字列"}
"""
import asyncio, logging, re
from datetime import datetime, timezone
from playwright.async_api import async_playwright

# ──────────────────────────────────────────
async def _async_post(title: str, body_html: str,
                      email: str, password: str) -> str:
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        pg  = await br.new_page()

        # 1) ログイン
        await pg.goto("https://note.com/login")
        await pg.fill('input[type="email"]', email)
        await pg.click('button[type="submit"]')
        await pg.fill('input[name="password"]', password)
        await pg.click('button[type="submit"]')
        await pg.wait_for_load_state("networkidle")

        # 2) 新規投稿
        await pg.goto("https://note.com/new")
        await pg.wait_for_selector('[placeholder="タイトル"]', timeout=15000)
        await pg.fill('[placeholder="タイトル"]', title)

        # 3) 本文 (iframe)
        iframe = pg.frame_locator('iframe').first
        await iframe.locator('div').click()
        # HTML を直接流し込み
        await iframe.locator('div').evaluate(
            "(el, html)=>{el.innerHTML = html}", body_html
        )

        # 4) 公開
        await pg.click('text=公開')
        await pg.click('text=投稿する')   # 確認モーダル
        await pg.wait_for_url(re.compile(r'https://note.com/.+/n.*'), timeout=30000)

        url = pg.url
        await br.close()
        return url

# ──────────────────────────────────────────
def post_to_note(title: str, body_html: str,
                 email: str, password: str) -> dict:
    """
    Returns:
        {"ok": True,  "url": "...", "posted_at": datetime}
        失敗時は {"ok": False, "error": "..."}
    """
    try:
        url = asyncio.run(_async_post(title, body_html, email, password))
        return {"ok": True, "url": url, "posted_at": datetime.now(timezone.utc)}
    except Exception as e:
        logging.exception(f"[NotePoster] 投稿失敗: {e}")
        return {"ok": False, "error": str(e)}
