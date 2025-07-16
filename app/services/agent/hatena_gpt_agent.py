# app/services/agents/hatena_gpt_agent.py

import logging
from playwright.async_api import async_playwright
from app.services.agent.base_agent import GPTAgentBase

logger = logging.getLogger(__name__)

async def run_hatena_signup(email: str, nickname: str, password: str):
    """はてなブログのアカウントをAIエージェントで作成"""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.hatena.ne.jp/register")  # ✅ はてなの登録ページ

        html = await page.content()
        goal = "はてなの会員登録フォームに必要な情報を入力して、アカウントを作成してください。"
        values = {
            "email": email,
            "nickname": nickname,
            "password": password
        }

        agent = GPTAgentBase(page, html, goal, values)
        await agent.run()

        await page.wait_for_timeout(5000)
        await browser.close()

        logger.info("[Hatena-Signup] 登録処理完了")
        return True
