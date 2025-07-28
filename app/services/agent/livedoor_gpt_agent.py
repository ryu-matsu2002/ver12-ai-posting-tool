import logging
import asyncio  # ✅ 追加
from app.services.blog_signup.livedoor_signup import signup

logger = logging.getLogger(__name__)


class LivedoorAgent:
    def __init__(self, site, email, password, nickname, token):
        self.site = site
        self.email = email
        self.password = password
        self.nickname = nickname
        self.token = token
        self.job_id = None

        # ✅ CAPTCHA処理に使う追加変数
        self._captcha_event = asyncio.Event()  # 解答入力まで待機するイベント
        self._captcha_solution = None  # 入力された解答文字列

    async def run(self) -> dict:
        logger.info("[LD-Agent] signup() を呼び出してアカウント作成を開始します")
        return await signup(
            site=self.site,
            email=self.email,
            password=self.password,
            nickname=self.nickname,
            token=self.token,
            job_id=self.job_id,
            agent=self  # ✅ signupに自身を渡す（後で使う）
        )

    # ✅ CAPTCHA解答を受け取ってイベント解除するメソッド
    def resume_with_captcha_solution(self, text: str):
        logger.info(f"[LD-Agent] CAPTCHA解答を受信: {text}")
        self._captcha_solution = text
        self._captcha_event.set()

    # ✅ CAPTCHA解答を取得するまで待機する
    async def wait_for_captcha_solution(self) -> str:
        logger.info("[LD-Agent] CAPTCHA解答待機中...")
        await self._captcha_event.wait()
        return self._captcha_solution
