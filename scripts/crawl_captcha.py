# scripts/crawl_captcha.py  ★上書きしてください

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

        # ① captcha-img が DOM に現れるまで待つ（visible でなくて OK）
        img = page.locator("#captcha-img")
        await img.wait_for(timeout=7_000)

        # ② src 属性を取得し完全 URL に変換
        src = await img.get_attribute("src")              # /register/captcha?123
        captcha_url = page.url.rstrip("/register/input") + src

        # ③ PNG バイト列を直接ダウンロード
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
