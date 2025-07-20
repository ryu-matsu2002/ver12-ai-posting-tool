import logging
from app.services.blog_signup.livedoor_signup import signup  # ✅ signup関数を利用

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
        logger.info("[LD-Agent] signup() を呼び出してアカウント作成を開始します")
        return await signup(
            site=self.site,
            email=self.email,
            password=self.password,
            nickname=self.nickname,
            token=self.token,
            job_id=self.job_id
        )
