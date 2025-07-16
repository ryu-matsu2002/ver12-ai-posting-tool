import asyncio
import logging
from playwright.async_api import async_playwright
from app.services.mail_utils.mail_gw import poll_latest_link_gw

logger = logging.getLogger(__name__)


class LivedoorAgent:
    def __init__(self, site, email, password, nickname, token):
        self.site = site
        self.email = email
        self.password = password
        self.nickname = nickname  # → ユーザーIDとして使う
        self.token = token
        self.job_id = None

    async def run(self) -> dict:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                logger.info("[LD-Agent] 🚀 Livedoor登録ページにアクセスします")
                await page.goto("https://member.livedoor.com/register/input", timeout=30_000)

                # ✅ セレクタが存在するまで待機して入力
                await page.wait_for_selector("#livedoor_id", timeout=10000)
                await page.fill("#livedoor_id", self.nickname)
                logger.info(f"[LD-Agent] 入力: livedoor_id = {self.nickname}")

                await page.fill("#password", self.password)
                await page.fill("#password2", self.password)
                logger.info(f"[LD-Agent] 入力: password (2回)")

                await page.fill("#email", self.email)
                logger.info(f"[LD-Agent] 入力: email = {self.email}")

                await asyncio.sleep(1.5)

                await page.click('input[type="submit"]')
                logger.info(f"[LD-Agent] 登録ボタンをクリック")

                await asyncio.sleep(3)

                content = await page.content()
                if "仮登録メールをお送りしました" not in content:
                    raise RuntimeError("仮登録が失敗した可能性があります")

                logger.info("[LD-Agent] ✅ 仮登録成功。メール認証を待機します...")

                # ✅ 認証リンク取得
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

                # ✅ 登録完了（APIキーは後で対応）
                api_key = "dummy-api-key"
                blog_id = self.nickname

                logger.info("[LD-Agent] 🎉 登録完了（仮）。APIキーは後続処理で設定")

                return {
                    "api_key": api_key,
                    "blog_id": blog_id,
                }

            except Exception as e:
                logger.error(f"[LD-Agent] エラー: {e}")
                raise

            finally:
                await browser.close()
