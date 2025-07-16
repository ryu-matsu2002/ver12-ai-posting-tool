# app/services/agent/livedoor_gpt_agent.py

import asyncio
import logging
from playwright.async_api import async_playwright
from app.services.ai_executor import ask_gpt_for_actions
from app.services.mail_utils.mail_gw import poll_latest_link_gw

logger = logging.getLogger(__name__)


class LivedoorAgent:
    def __init__(self, site, email, password, nickname, token):
        self.site = site
        self.email = email
        self.password = password
        self.nickname = nickname
        self.token = token
        self.job_id = None  # 任意

    async def run(self) -> dict:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                logger.info("[LD-GPT-Agent] 🚀 Livedoor登録ページにアクセスします")
                await page.goto("https://member.livedoor.com/register/input", timeout=30_000)

                # ✅ HTMLと目標をGPTに渡して、実行手順を取得
                html = await page.content()
                actions = await ask_gpt_for_actions(
                    html=html,
                    goal="Livedoorブログに新規登録する",
                    values={
                        "email": self.email,
                        "password": self.password,
                        "nickname": self.nickname,
                    },
                )

                # ✅ GPTの指示を順に実行
                for step in actions:
                    action = step["action"]
                    selector = step["selector"]
                    value = step.get("value")

                    if action == "fill":
                        real_value = {
                            "EMAIL": self.email,
                            "PASSWORD": self.password,
                            "NICKNAME": self.nickname,
                        }.get(value, value)
                        await page.fill(selector, real_value)
                        logger.info(f"[LD-GPT-Agent] 入力: {selector} = {real_value}")

                    elif action == "click":
                        try:
                            await page.wait_for_selector(selector, timeout=10000)
                            await page.click(selector)
                            logger.info(f"[LD-GPT-Agent] クリック: {selector}")
                        except Exception as e:
                            logger.error(f"[LD-GPT-Agent] ❌ クリック失敗: {selector} - {e}")
                            raise

                    await asyncio.sleep(1.5)

                await asyncio.sleep(3)

                # ✅ 仮登録の成功確認
                content = await page.content()
                if "仮登録メールをお送りしました" not in content:
                    raise RuntimeError("仮登録が失敗した可能性があります")

                logger.info("[LD-GPT-Agent] ✅ 仮登録成功。メール認証を待機します...")

                # ✅ メールから認証リンク取得（← token が必要）
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-GPT-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

                # ✅ APIキーとBlog IDを返す（ダミー）
                api_key = "dummy-api-key"
                blog_id = self.nickname

                logger.info("[LD-GPT-Agent] 🎉 登録完了（仮）。APIキーは後続処理で設定")

                return {
                    "api_key": api_key,
                    "blog_id": blog_id,
                }

            except Exception as e:
                logger.error(f"[LD-GPT-Agent] エラー: {e}")
                raise

            finally:
                await browser.close()


# 旧関数形式の互換：run_livedoor_signup()
async def run_livedoor_signup(site, email, token, nickname, password, job_id=None):
    agent = LivedoorAgent(
        site=site,
        email=email,
        password=password,
        nickname=nickname,
        token=token  # ✅ 修正点：tokenを渡す
    )
    agent.job_id = job_id
    return await agent.run()
