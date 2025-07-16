import asyncio
import logging
from playwright.async_api import async_playwright
from app.services.mail_utils.mail_gw import poll_latest_link_gw
from app.services.captcha_solver import solve  # ✅ CAPTCHA解読

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

                # ✅ セレクタ取得の保護（ログ出力つき）
                try:
                    logger.info("[LD-Agent] livedoor_id のセレクタ取得を試みます")
                    await page.wait_for_selector("#livedoor_id", timeout=10000)
                except Exception as selector_error:
                    html = await page.content()
                    logger.error(f"[LD-Agent] livedoor_id セレクタ取得に失敗: {selector_error}")
                    logger.warning(f"[LD-Agent][DEBUG] HTML Snapshot:\n{html[:1000]}")
                    await page.screenshot(path="/tmp/ld_id_fail.png", full_page=True)
                    raise

                logger.info(f"[LD-Agent] livedoor_id 入力: {self.nickname}")
                await page.fill("#livedoor_id", self.nickname)

                logger.info("[LD-Agent] パスワード入力")
                await page.fill("#password", self.password)

                logger.info("[LD-Agent] パスワード（確認）入力")
                await page.fill("#password2", self.password)

                logger.info(f"[LD-Agent] メールアドレス入力: {self.email}")
                await page.fill("#email", self.email)

                await asyncio.sleep(1.5)

                # ✅ 登録ボタン状態確認（＋デバッグ情報出力）
                logger.info("[LD-Agent] 登録ボタンの状態確認開始")
                try:
                    await page.wait_for_selector('input[type="submit"]', timeout=10000)
                    visible = await page.is_visible('input[type="submit"]')
                    enabled = await page.is_enabled('input[type="submit"]')
                    logger.info(f"[LD-Agent] 登録ボタン: visible={visible}, enabled={enabled}")
                except Exception as submit_check_err:
                    logger.error(f"[LD-Agent] 登録ボタンの確認に失敗: {submit_check_err}")
                    try:
                        html = await page.content()
                        logger.warning(f"[LD-Agent][DEBUG] submitボタン取得失敗時HTML:\n{html[:1000]}")
                        await page.screenshot(path="/tmp/ld_submit_fail.png", full_page=True)
                        logger.warning("[LD-Agent][DEBUG] スクリーンショット保存済み: /tmp/ld_submit_fail.png")
                    except Exception as e:
                        logger.warning(f"[LD-Agent][DEBUG] スクショまたはHTML保存失敗: {e}")
                    raise

                # ✅ 登録ボタンをクリック（または fallback）
                try:
                    await page.eval_on_selector('input[type="submit"]', "el => el.scrollIntoView()")
                    await asyncio.sleep(0.5)
                    await page.click('input[type="submit"]')
                    logger.info("[LD-Agent] 登録ボタンをクリック")
                except Exception as e:
                    logger.warning(f"[LD-Agent] submitボタンのクリックに失敗、form.submit() に切り替え: {e}")
                    await page.eval_on_selector('form[action="/register/input"]', "form => form.submit()")

                # ✅ CAPTCHAポップアップ待機
                await page.wait_for_selector('img[src*="/captcha"]', timeout=10000)
                captcha_url = await page.get_attribute('img[src*="/captcha"]', "src")
                logger.info(f"[LD-Agent] CAPTCHA画像URL: {captcha_url}")
                img_response = await page.request.get(f"https://member.livedoor.com{captcha_url}")
                img_bytes = await img_response.body()

                captcha_text = solve(img_bytes)
                logger.info(f"[LD-Agent] CAPTCHA判定結果: {captcha_text}")

                await page.fill('input[name="captcha_value"]', captcha_text)
                logger.info("[LD-Agent] CAPTCHAを入力完了")
                await asyncio.sleep(1)

                try:
                    html = await page.content()
                    logger.warning(f"[LD-Agent][DEBUG] CAPTCHA送信直前のHTML:\n{html[:1000]}")
                    await page.screenshot(path="/tmp/ld_captcha_screen.png", full_page=True)
                    logger.warning("[LD-Agent][DEBUG] スクリーンショット保存済み: /tmp/ld_captcha_screen.png")
                except Exception as debug_e:
                    logger.warning(f"[LD-Agent][DEBUG] スクリーンショットまたはHTML取得に失敗: {debug_e}")

                # ✅ 完了ボタン処理
                try:
                    await page.wait_for_selector('button:has-text("完了")', timeout=15000)
                    is_visible = await page.is_visible('button:has-text("完了")')
                    is_enabled = await page.is_enabled('button:has-text("完了")')
                    logger.info(f"[LD-Agent] 完了ボタン: visible={is_visible}, enabled={is_enabled}")

                    if is_visible and is_enabled:
                        await page.click('button:has-text("完了")')
                        logger.info("[LD-Agent] 完了ボタンをクリック")
                    else:
                        raise Exception("完了ボタンが無効 or 非表示")

                except Exception as click_error:
                    logger.warning(f"[LD-Agent] 完了ボタンクリック失敗: {click_error}")
                    logger.info("[LD-Agent] form.submit() を試行")
                    await page.eval_on_selector('form[action="/register/confirm"]', "form => form.submit()")

                await asyncio.sleep(2)
                content = await page.content()
                if "ご登録ありがとうございます" not in content:
                    raise RuntimeError("登録完了画面が表示されませんでした")

                logger.info("[LD-Agent] ✅ 登録成功、メール認証を待機します")

                # ✅ メール認証リンク取得
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

                # ✅ 登録完了（仮APIキー）
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
