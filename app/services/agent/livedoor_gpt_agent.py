import asyncio
import logging
from playwright.async_api import async_playwright
from app.services.mail_utils.mail_gw import poll_latest_link_gw
from app.services.captcha_solver import solve
from app.services.captcha_solver.save_failed import save_failed_captcha_image

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
                await page.goto("https://member.livedoor.com/register/input", timeout=30000)

                # フォーム入力
                await page.wait_for_selector("#livedoor_id", timeout=10000)
                logger.info(f"[LD-Agent] livedoor_id 入力: {self.nickname}")
                await page.fill("#livedoor_id", self.nickname)

                logger.info("[LD-Agent] パスワード入力")
                await page.fill("#password", self.password)
                await page.fill("#password2", self.password)

                logger.info(f"[LD-Agent] メールアドレス入力: {self.email}")
                await page.fill("#email", self.email)

                # 登録ボタン押下 → CAPTCHAポップアップが出る
                logger.info("[LD-Agent] 登録ボタンをクリックします")
                await page.click('input[value="ユーザー情報を登録"]')
                await asyncio.sleep(2)

                # CAPTCHAポップアップを待機
                logger.info("[LD-Agent] CAPTCHAポップアップを検出します")
                await page.wait_for_selector("img[src^='/register/captcha']", timeout=15000)
                captcha_img_selector = "img[src^='/register/captcha']"
                captcha_input_selector = 'input[type="text"]'
                complete_button_selector = '#commit-button'

                # CAPTCHA画像取得と解読
                captcha_url = await page.get_attribute(captcha_img_selector, "src")
                logger.info(f"[LD-Agent] CAPTCHA画像URL: {captcha_url}")
                img_response = await page.request.get(f"https://member.livedoor.com{captcha_url}")
                img_bytes = await img_response.body()

                captcha_text = solve(img_bytes)
                logger.info(f"[LD-Agent] CAPTCHA解読結果: {captcha_text}")
                await page.fill(captcha_input_selector, captcha_text)

                # CAPTCHAスクリーンショット（前）
                await page.screenshot(path="/tmp/ld_captcha_screen.png", full_page=True)

                # 「完了」ボタンをクリック
                await page.wait_for_selector(complete_button_selector, timeout=10000)
                await page.click(complete_button_selector)
                logger.info("[LD-Agent] CAPTCHA完了ボタンをクリック")

                # 完了後の確認
                await asyncio.sleep(2)
                content = await page.content()
                current_url = page.url

                fail_patterns = ["正しくありません", "認証コードが間違っています", "入力し直してください"]
                if any(pat in content for pat in fail_patterns):
                    save_failed_captcha_image("/tmp/ld_captcha_screen.png", reason="captcha_fail")
                    await page.screenshot(path="/tmp/ld_captcha_failed.png", full_page=True)
                    logger.error("[LD-Agent] ❌ CAPTCHA失敗と判定されました")
                    raise RuntimeError("CAPTCHA認証に失敗しました")

                success_patterns = [
                    "ご登録ありがとうございます",
                    "メールを送信しました",
                    "/register/done"
                ]
                if not any(pat in content or pat in current_url for pat in success_patterns):
                    await page.screenshot(path="/tmp/ld_registration_incomplete.png", full_page=True)
                    logger.warning("[LD-Agent] ❌ 登録成功の痕跡が見つかりません")
                    raise RuntimeError("登録完了画面が表示されませんでした")

                logger.info("[LD-Agent] ✅ CAPTCHA突破・登録成功")

                # メール認証リンク取得
                verification_url = None
                try:
                    async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                        verification_url = link
                        break
                except Exception as poll_err:
                    await page.screenshot(path="/tmp/ld_verification_poll_fail.png", full_page=True)
                    logger.error(f"[LD-Agent] 認証リンク取得中にエラー: {poll_err}")
                    raise

                if not verification_url:
                    await page.screenshot(path="/tmp/ld_verification_url_none.png", full_page=True)
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-Agent] 認証リンクへアクセス: {verification_url}")
                await page.goto(verification_url, timeout=30000)
                await asyncio.sleep(2)

                logger.info("[LD-Agent] 🎉 登録完了（仮）")

                return {
                    "api_key": "dummy-api-key",
                    "blog_id": self.nickname
                }

            except Exception as e:
                logger.error(f"[LD-Agent] エラー: {e}")
                raise

            finally:
                await browser.close()
