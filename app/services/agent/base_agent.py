# app/services/agent/base_agent.py

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BlogAgent(ABC):
    def __init__(self, site, email, password, nickname):
        self.site = site  # Siteオブジェクト（SQLAlchemyなど）
        self.email = email
        self.password = password
        self.nickname = nickname

    @abstractmethod
    async def run(self):
        """エージェント本体処理。サブクラスで実装。"""
        raise NotImplementedError("BlogAgent.run() はオーバーライド必須です")

    async def log_info(self, message: str):
        logger.info(f"[Agent:{self.__class__.__name__}] {message}")

    async def log_error(self, message: str):
        logger.error(f"[Agent:{self.__class__.__name__}] {message}")
