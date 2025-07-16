import asyncio
import logging
import random
import string
from playwright.async_api import async_playwright
from app.services.mail_utils.mail_gw import poll_latest_link_gw
from app.services.captcha_solver import solve

logger = logging.getLogger(__name__)

def ensure_valid_livedoor_id(nickname: str) -> str:
    nickname = ''.join(c for c in nickname if c.isalnum())[:20]
    while len(nickname) < 3:
        nickname += random.choice(string.ascii_lowercase + string.digits)
    if not nickname[0].isalpha():
        nickname = random.choice(string.ascii_lowercase) + nickname[1:]
    return nickname


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
                self.nickname = ensure_valid_livedoor_id(self.nickname)
                logger.info(f"[LD-Agent] livedoor_id 入力: {self.nickname}")
                await page.fill("#livedoor_id", self.nickname)

                logger.info("[LD-Agent] パスワード入力")
                await page.fill("#password", self.password)
                await page.fill("#password2", self.password)

                logger.info(f"[LD-Agent] メールアドレス入力: {self.email}")
                await page.fill("#email", self.email)
                await asyncio.sleep(1.5)

                # CAPTCHA画像取得と認識
                await page.wait_for_selector("#captcha-img", timeout=10000)
                captcha_url = await page.get_attribute("#captcha-img", "src")
                logger.info(f"[LD-Agent] CAPTCHA画像URL: {captcha_url}")
                img_response = await page.request.get(f"https://member.livedoor.com{captcha_url}")
                img_bytes = await img_response.body()

                captcha_text = solve(img_bytes)
                logger.info(f"[LD-Agent] CAPTCHA判定結果: {captcha_text}")
                await page.fill("#captcha", captcha_text)
                await asyncio.sleep(1)

                # CAPTCHA直後スクショ
                try:
                    await page.screenshot(path="/tmp/ld_captcha_screen.png", full_page=True)
                    logger.warning("[LD-Agent][DEBUG] CAPTCHA送信直前スクリーンショット保存済み")
                except Exception as debug_e:
                    logger.warning(f"[LD-Agent][DEBUG] CAPTCHA直前のスクショ取得失敗: {debug_e}")

                # 完了ボタン処理（submit）
                try:
                    await page.wait_for_selector("#commit-button", timeout=15000)
                    is_visible = await page.is_visible("#commit-button")
                    is_enabled = await page.is_enabled("#commit-button")
                    logger.info(f"[LD-Agent] commit-button visible={is_visible}, enabled={is_enabled}")

                    if is_visible and is_enabled:
                        await page.click("#commit-button")
                        logger.info("[LD-Agent] 完了ボタンをクリック")
                    else:
                        raise Exception("commit-button が無効 or 非表示")
                except Exception as click_error:
                    logger.warning(f"[LD-Agent] commit-buttonクリック失敗: {click_error}")
                    await page.eval_on_selector('form[action="/register/confirm"]', "form => form.submit()")

                await asyncio.sleep(2)
                content = await page.content()
                if "ご登録ありがとうございます" not in content:
                    raise RuntimeError("登録完了画面が表示されませんでした")

                logger.info("[LD-Agent] ✅ 登録成功、メール認証を待機します")

                # メール認証リンクを取得
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

                # ダミーAPI情報（後続で正式に設定）
                api_key = "dummy-api-key"
                blog_id = self.nickname

                logger.info("[LD-Agent] 🎉 登録完了。仮APIキーとブログIDを返却")

                return {
                    "api_key": api_key,
                    "blog_id": blog_id,
                }

            except Exception as e:
                logger.error(f"[LD-Agent] エラー: {e}")
                raise

            finally:
                await browser.close()
