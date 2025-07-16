# app/services/agents/livedoor_gpt_agent.py

import asyncio
import logging
from playwright.async_api import async_playwright

from app.services.ai_executor import ask_gpt_for_actions

logger = logging.getLogger(__name__)

async def run_livedoor_signup(site, email, token, nickname, password) -> dict:
    """
    GPTを使ってLivedoorブログの登録を自動で行う。
    Returns: dict { blog_id: ..., api_key: ... }
    """

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
                    "email": email,
                    "password": password,
                    "nickname": nickname,
                },
            )

            # ✅ GPTから得た指示を実行
            for step in actions:
                action = step["action"]
                selector = step["selector"]
                value = step.get("value")

                if action == "fill":
                    real_value = {
                        "EMAIL": email,
                        "PASSWORD": password,
                        "NICKNAME": nickname,
                    }.get(value, value)
                    await page.fill(selector, real_value)
                    logger.info(f"[LD-GPT-Agent] 入力: {selector} = {real_value}")

                elif action == "click":
                    await page.click(selector)
                    logger.info(f"[LD-GPT-Agent] クリック: {selector}")

                await asyncio.sleep(1.5)  # 各操作の間に少し待機

            await asyncio.sleep(3)

            # ✅ 登録成功かどうかをHTMLで判定（暫定）
            content = await page.content()
            if "仮登録メールをお送りしました" not in content:
                raise RuntimeError("仮登録が失敗した可能性があります")

            logger.info("[LD-GPT-Agent] ✅ 仮登録成功。メール認証を待機します...")

            # ✅ メールから認証リンクを取得
            from app.services.mail_utils.mail_gw import poll_latest_link_gw
            verification_url = None
            async for link in poll_latest_link_gw(token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                verification_url = link
                break

            if not verification_url:
                raise RuntimeError("認証リンクの取得に失敗しました")

            logger.info(f"[LD-GPT-Agent] 認証リンクへ移動: {verification_url}")
            await page.goto(verification_url, timeout=30_000)
            await asyncio.sleep(2)

            # ✅ APIキー抽出（暫定）
            api_key = "dummy-api-key"
            blog_id = nickname
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

# app/services/agent/livedoor_gpt_agent.py

class LivedoorAgent:
    def __init__(self, site):
        self.site = site

    async def run(self):
        print(f"[仮実行] LivedoorAgent 実行中（site_id={self.site.id}）")
        return {"status": "not_implemented"}
