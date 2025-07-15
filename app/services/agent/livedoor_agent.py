# app/services/agent/livedoor_agent.py

from .base_agent import BlogAgent
from app.services.mail_utils.mail_gw import create_inbox, poll_latest_link_gw
from app.services.captcha_solver import solve  # ← 独自AI CAPTCHA解読
from pathlib import Path
import asyncio
from playwright.async_api import async_playwright
from app.services.agent import agent_logger  # ← ✅ ログ出力用

SUCCESS_PATTERNS = [
    "メールを送信しました",
    "仮登録",
    "仮登録メールをお送りしました"
]

class LivedoorAgent(BlogAgent):
    async def run(self):
        await self.log_info("Livedoorアカウント登録エージェントを起動")

        # ✅ ステップログ：ジョブ開始
        agent_logger.log(
            job_id=self.job_id,
            step="start",
            message="Livedoor 登録エージェントを開始"
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                # 1. 登録ページへアクセス
                await page.goto("https://member.livedoor.com/register/input", timeout=20000)
                await self.log_info("登録ページにアクセスしました")
                agent_logger.log(self.job_id, "navigate", "登録ページにアクセス完了")

                # 2. 入力
                await page.fill("#mail", self.email)
                await page.fill("#password", self.password)
                await page.fill("#username", self.nickname)
                agent_logger.log(self.job_id, "fill", f"入力完了（{self.email} / {self.nickname}）")

                # 3. CAPTCHA画像を保存しAIで解読
                captcha_path = "/tmp/captcha.png"
                captcha_elem = await page.query_selector("#captcha_img")
                await captcha_elem.screenshot(path=captcha_path)
                solved = solve(captcha_path)
                await self.log_info(f"CAPTCHA解読結果: {solved}")
                agent_logger.log(self.job_id, "captcha", f"CAPTCHA解読結果：{solved}")
                await page.fill("#captcha_text", solved)

                # 4. 送信
                await page.click("#submit-button")
                await asyncio.sleep(3)
                agent_logger.log(self.job_id, "submit", "フォーム送信を実行")

                # 5. 成否判定
                content = await page.content()
                Path("/tmp/ld_signup_post_submit.html").write_text(content, encoding="utf-8")

                if not any(pat in content for pat in SUCCESS_PATTERNS):
                    await self.log_error("登録失敗：成功メッセージが見つかりません")
                    agent_logger.log(self.job_id, "error", "登録失敗：成功メッセージが存在しません")
                    raise RuntimeError("Livedoor登録に失敗しました")

                await self.log_info("フォーム送信成功 → 認証メール待機へ")
                agent_logger.log(self.job_id, "submitted", "フォーム送信成功、メール認証リンクを待機")

                # 6. メール認証リンクを待機
                token = await create_inbox(self.email)
                link = None
                async for l in poll_latest_link_gw(token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    link = l
                    break

                if not link:
                    await self.log_error("認証リンクが取得できませんでした")
                    agent_logger.log(self.job_id, "error", "認証リンクの取得に失敗")
                    raise RuntimeError("メール認証リンク取得失敗")

                await page.goto(link, timeout=30000)
                await self.log_info(f"メール認証完了：{link}")
                agent_logger.log(self.job_id, "verified", f"メール認証完了：{link}")

            finally:
                await browser.close()
                agent_logger.log(self.job_id, "finished", "Livedoor登録処理が完了し、ブラウザを閉じました")

# livedoor_agent.py の末尾に以下を追加
async def run_livedoor_signup(job_id: int, email: str, password: str, nickname: str):
    agent = LivedoorAgent(job_id=job_id, email=email, password=password, nickname=nickname)
    await agent.run()
