# app/services/agents/base_agent.py

import logging
from abc import ABC, abstractmethod
from playwright.async_api import Page
from app.services.ai_executor import ask_gpt_for_actions

logger = logging.getLogger(__name__)

class GPTAgentBase(ABC):
    def __init__(self, page: Page, html: str, goal: str, values: dict):
        self.page = page
        self.html = html
        self.goal = goal
        self.values = values

    async def run(self):
        """
        GPTにHTMLと目標・値を渡して、指示を受け取り、ページ上で実行する
        """
        logger.info(f"[GPTAgent] 🎯 目標: {self.goal}")

        actions = await ask_gpt_for_actions(
            html=self.html,
            goal=self.goal,
            values=self.values
        )

        for step in actions:
            action = step.get("action")
            selector = step.get("selector")
            value = step.get("value", "")

            try:
                if action == "fill":
                    actual_value = self.values.get(value, value)
                    await self.page.fill(selector, actual_value)
                    logger.info(f"[GPTAgent] 入力: {selector} = {actual_value}")

                elif action == "click":
                    await self.page.click(selector)
                    logger.info(f"[GPTAgent] クリック: {selector}")

                await self.page.wait_for_timeout(1500)

            except Exception as e:
                logger.warning(f"[GPTAgent] ⚠️ 実行失敗: {action} {selector}: {e}")
