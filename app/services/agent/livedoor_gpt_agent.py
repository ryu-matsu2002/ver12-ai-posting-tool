import asyncio
import logging
from playwright.async_api import async_playwright
from app.services.mail_utils.mail_gw import poll_latest_link_gw
from app.services.captcha_solver import solve  # ✅ 追加

logger = logging.getLogger(__name__)


class LivedoorAgent:
    def __init__(self, site, email, password, nickname, token):
        self.site = site
        self.email = email
        self.password = password
        self.nickname = nickname
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

                # 基本情報入力
                await page.wait_for_selector("#livedoor_id", timeout=10000)
                await page.fill("#livedoor_id", self.nickname)
                await page.fill("#password", self.password)
                await page.fill("#password2", self.password)
                await page.fill("#email", self.email)
                await asyncio.sleep(1.5)

                await page.click('input[type="submit"]')  # 登録ボタン
                logger.info("[LD-Agent] ✅ 登録ボタンクリック後、CAPTCHAを待機")

                # CAPTCHAウィンドウを検出して待機
                await page.wait_for_selector("#captcha-img", timeout=10000)

                # 画像データを取得
                captcha_url = await page.get_attribute("#captcha-img", "src")
                logger.info(f"[LD-Agent] CAPTCHA画像URL: {captcha_url}")
                img_response = await page.request.get(f"https://member.livedoor.com{captcha_url}")
                img_bytes = await img_response.body()

                # CAPTCHA推論
                captcha_text = solve(img_bytes)
                logger.info(f"[LD-Agent] CAPTCHA判定結果: {captcha_text}")

                # 入力して「完了」ボタンを押す
                await page.fill("#captcha", captcha_text)
                await asyncio.sleep(1)

                html = await page.content()
                logger.warning(f"[LD-Agent][DEBUG] CAPTCHA送信直前のHTML:\n{html[:1000]}")
                await page.screenshot(path="/tmp/ld_captcha_screen.png", full_page=True)
                logger.warning("[LD-Agent][DEBUG] スクリーンショット保存済み: /tmp/ld_captcha_screen.png")

                await page.wait_for_selector("#commit-button", timeout=15000)
                await page.click("#commit-button")
                logger.info("[LD-Agent] 完了ボタンをクリック")

                # 仮登録成功判定（2枚目の画面）
                await asyncio.sleep(2)
                content = await page.content()
                if "ご登録ありがとうございます" not in content:
                    raise RuntimeError("登録完了画面が表示されませんでした")

                logger.info("[LD-Agent] ✅ 登録成功、メール認証を待機します")

                # メール認証リンクの取得
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

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
