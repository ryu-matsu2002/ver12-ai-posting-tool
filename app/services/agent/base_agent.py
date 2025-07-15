# app/services/agent/base_agent.py

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BlogAgent(ABC):
    """
    すべての外部SEO用AIエージェントの基底クラス
    """

    def __init__(self, site, email, password, nickname):
        self.site = site                  # Siteモデル（SQLAlchemy）
        self.email = email                # 登録用メールアドレス
        self.password = password          # 任意生成 or 固定パス
        self.nickname = nickname          # 表示用ニックネーム

        self.success = False              # 成功フラグ
        self.log = []                     # 実行ログを蓄積（UI表示用にも）

    def log_step(self, msg):
        """
        処理の進行状況を記録し、ログにも出力
        """
        self.log.append(msg)
        logger.info(f"[Agent] {msg}")

    @abstractmethod
    async def run(self):
        """
        具象エージェントが必ず実装する非同期メイン処理
        """
        pass
