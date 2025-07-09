# scripts/crawl_captcha.py

import asyncio
import time
from pathlib import Path
from uuid import uuid4

from playwright.async_api import async_playwright

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "https://member.livedoor.com/register/input"


async def crawl_once(p, index: int):
    browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = await browser.new_page()
    try:
        await page.goto(TARGET, timeout=30_000)

        # ① CAPTCHA画像が DOM に現れるのを待つ（hidden でもOK）
        await page.wait_for_selector("#captcha-img", state="attached", timeout=7000)
        img = page.locator("#captcha-img")

        # ② src 属性を取得し、null チェック
        src = await img.get_attribute("src")
        if not src:
            print(f"⚠️  CAPTCHA src not found at index={index}")
            return

        # ③ 完全URLにして PNG バイト列を取得
        base_url = page.url.split("/register/input")[0]
        captcha_url = base_url + src

        resp = await page.request.get(captcha_url, timeout=15_000)
        if resp.ok:
            img_bytes = await resp.body()
            filename = OUTPUT_DIR / f"{index}_{uuid4().hex}.png"
            filename.write_bytes(img_bytes)
            print(f"✅ saved {filename.name}")
        else:
            print(f"⚠️  HTTP {resp.status} for {captcha_url}")

    except Exception as e:
        print(f"❌ failed at {index}: {e}")
    finally:
        await browser.close()


async def main():
    async with async_playwright() as p:
        for i in range(300):          # 必要なら枚数を増減
            await crawl_once(p, i)
            await asyncio.sleep(1.2)  # 人間らしい待機

if __name__ == "__main__":
    asyncio.run(main())
