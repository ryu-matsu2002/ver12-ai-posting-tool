# scripts/crawl_captcha.py

import asyncio
import time
from pathlib import Path
from uuid import uuid4

from playwright.async_api import async_playwright

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_once(playwright, index):
    browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = await browser.new_page()
    try:
        await page.goto("https://member.livedoor.com/register/input", timeout=30_000)
        captcha_locator = page.locator("img[src*='captcha']")
        await captcha_locator.wait_for(timeout=5000)
        img_bytes = await captcha_locator.screenshot()
        filename = OUTPUT_DIR / f"{index}_{uuid4().hex}.png"
        filename.write_bytes(img_bytes)
        print(f"✅ {filename.name}")
    except Exception as e:
        print(f"❌ failed at {index}: {e}")
    finally:
        await browser.close()


async def main():
    async with async_playwright() as p:
        for i in range(300):  # 1 回で 300 枚（必要に応じて増減可）
            await crawl_once(p, i)
            await asyncio.sleep(1.0 + (i % 3) * 0.5)  # 人間らしい間隔

if __name__ == "__main__":
    asyncio.run(main())
