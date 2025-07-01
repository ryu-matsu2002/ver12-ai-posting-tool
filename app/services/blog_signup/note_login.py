import asyncio, logging
from playwright.async_api import async_playwright

async def _login(email: str, password: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context()
        page    = await ctx.new_page()
        await page.goto("https://note.com/login")
        await page.fill('input[type="email"]', email)
        await page.click('button[type="submit"]')
        await page.fill('input[name="password"]', password)
        await page.click('button[type="submit"]')
        await page.wait_for_load_state("networkidle")
        cookies = await ctx.cookies("https://note.com")
        await browser.close()
        return cookies

def get_note_cookies(email: str, password: str):
    try:
        return asyncio.run(_login(email, password))
    except Exception as e:
        logging.exception(f"[NoteLogin] 失敗: {e}")
        return []
