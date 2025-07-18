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
                await page.goto("https://member.livedoor.com/register/input", timeout=30_000)

                await page.wait_for_selector("#livedoor_id", timeout=10000)
                logger.info(f"[LD-Agent] livedoor_id 入力: {self.nickname}")
                await page.fill("#livedoor_id", self.nickname)

                logger.info("[LD-Agent] パスワード入力")
                await page.fill("#password", self.password)

                logger.info("[LD-Agent] パスワード（確認）入力")
                await page.fill("#password2", self.password)

                logger.info(f"[LD-Agent] メールアドレス入力: {self.email}")
                await page.fill("#email", self.email)
                await asyncio.sleep(1.5)

                logger.info("[LD-Agent] 登録ボタンの状態確認開始")
                await page.wait_for_selector('input[value="ユーザー情報を登録"]', timeout=10000)
                visible = await page.is_visible('input[type="submit"]')
                enabled = await page.is_enabled('input[type="submit"]')
                logger.info(f"[LD-Agent] 登録ボタン: visible={visible}, enabled={enabled}")

                await page.eval_on_selector('input[type="submit"]', "el => el.scrollIntoView()")
                await asyncio.sleep(0.5)
                await page.click('input[value="ユーザー情報を登録"]')
                logger.info("[LD-Agent] 登録ボタンをクリック")

                # CAPTCHA処理
                await page.wait_for_selector("#captcha-img", timeout=10000)
                captcha_url = await page.get_attribute("#captcha-img", "src")
                logger.info(f"[LD-Agent] CAPTCHA画像URL: {captcha_url}")
                img_response = await page.request.get(f"https://member.livedoor.com{captcha_url}")
                img_bytes = await img_response.body()

                captcha_text = solve(img_bytes)
                logger.info(f"[LD-Agent] CAPTCHA判定結果: {captcha_text}")
                await page.fill("#captcha", captcha_text)
                logger.info("[LD-Agent] CAPTCHAを入力完了")
                await asyncio.sleep(1)

                await page.screenshot(path="/tmp/ld_captcha_screen.png", full_page=True)
                html_before = await page.content()
                logger.warning(f"[LD-Agent][DEBUG] CAPTCHA送信直前のHTML:\n{html_before[:1000]}")

                await page.wait_for_selector("#commit-button", timeout=15000)
                if await page.is_visible("#commit-button") and await page.is_enabled("#commit-button"):
                    await page.click("#commit-button")
                    logger.info("[LD-Agent] 完了ボタンをクリック")
                else:
                    raise Exception("commit-button が無効 or 非表示")

                # CAPTCHA送信後の検出
                await asyncio.sleep(2)
                content = await page.content()
                current_url = page.url

                # ✅ CAPTCHA失敗検出
                captcha_fail_patterns = ["正しくありません", "再度入力", "認証コードが間違っています", "入力し直してください"]
                if any(pat in content for pat in captcha_fail_patterns):
                    await page.screenshot(path="/tmp/ld_captcha_fail_detected.png", full_page=True)
                    save_failed_captcha_image("/tmp/ld_captcha_screen.png", reason="bad_prediction")
                    logger.error("[LD-Agent] CAPTCHA入力が失敗した可能性があります")
                    raise RuntimeError("CAPTCHA認証に失敗した可能性があります")

                # ✅ 登録成功検出強化
                success_patterns = [
                    "ご登録ありがとうございます",
                    "メールを送信しました",
                    "/register/done"
                ]
                if not any(pat in content or pat in current_url for pat in success_patterns):
                    await page.screenshot(path="/tmp/ld_registration_incomplete.png", full_page=True)
                    await page.screenshot(path="/tmp/ld_post_submit_debug.png", full_page=True)
                    save_failed_captcha_image("/tmp/ld_captcha_screen.png", reason="submit_fail")
                    logger.warning(f"[LD-Agent][DEBUG] 登録失敗時のHTML:\n{content[:1000]}")
                    raise RuntimeError("登録完了画面が表示されませんでした")

                logger.info("[LD-Agent] ✅ 登録成功、メール認証を待機します")

                # 認証リンク取得
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
