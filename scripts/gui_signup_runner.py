import json
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from app.services.mail_utils.mail_gw import poll_latest_link_gw
from playwright.async_api import async_playwright

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main(input_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 入力ファイルを読み込む
    input_data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    email = input_data["email"]
    token = input_data["token"]
    nickname = input_data["nickname"]
    password = input_data["password"]
    output_path = Path(input_data["output_path"])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        page = await browser.new_page()

        try:
            # Step1: ユーザー情報自動入力
            await page.goto("https://member.livedoor.com/register/input")
            await page.fill('input[name="livedoor_id"]', nickname)
            await page.fill('input[name="password"]', password)
            await page.fill('input[name="password2"]', password)
            await page.fill('input[name="email"]', email)

            # CAPTCHAページへ遷移
            await page.click('input[value="ユーザー情報を登録"]')
            print("🧠 CAPTCHA入力画面に遷移しました。手動で突破してください。")

            # CAPTCHA突破完了まで待機
            await page.wait_for_url("**/register/done", timeout=300000)
            print("✅ CAPTCHA突破が成功しました。登録完了画面に遷移しています。")

            # Step2: メール認証リンクを取得
            logger.info("[GUI-RUNNER] メール確認中...")
            url = None
            for i in range(3):
                url = await poll_latest_link_gw(token)
                if url:
                    break
                logger.warning(f"[GUI-RUNNER] メールリンクが取得できません（試行{i+1}/3）")
                await asyncio.sleep(5)

            if not url:
                raise RuntimeError("確認メールリンクが取得できません（最大リトライ）")

            await page.goto(url)
            await page.wait_for_timeout(2000)

            # Step3: API Key抽出
            html = await page.content()
            blog_id = await page.input_value("#livedoor_blog_id")
            api_key = await page.input_value("#atompub_key")

            if not blog_id or not api_key:
                fail_html = f"/tmp/ld_gui_final_fail_{timestamp}.html"
                fail_png = f"/tmp/ld_gui_final_fail_{timestamp}.png"
                Path(fail_html).write_text(html, encoding="utf-8")
                await page.screenshot(path=fail_png)
                raise RuntimeError("API KeyまたはBlog IDが取得できません")

            # 出力ファイルに保存
            output_path.write_text(json.dumps({
                "blog_id": blog_id,
                "api_key": api_key
            }), encoding="utf-8")

            logger.info(f"[GUI-RUNNER] 完了: blog_id={blog_id}")

        finally:
            await browser.close()

# エントリーポイント
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gui_signup_runner.py <input_json_path>")
        sys.exit(1)

    input_json = sys.argv[1]
    asyncio.run(main(input_json))
