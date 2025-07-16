# app/services/agent/livedoor_gpt_agent.py

import asyncio
import logging
import re
from playwright.async_api import async_playwright

from app.services.mail_utils.mail_gw import poll_latest_link_gw
from app.utils.html_utils import extract_hidden_inputs

logger = logging.getLogger(__name__)

class LivedoorAgent:
    def __init__(self, site, email, token, nickname, password):
        self.site = site
        self.email = email
        self.token = token
        self.nickname = nickname
        self.password = password
        self.job_id = None  # あとでログに使用可能

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                logger.info("[LD-GPT-Agent] 🚀 Livedoor登録ページにアクセスします")
                await page.goto("https://member.livedoor.com/register/input", timeout=30_000)

                # ✅ hidden input取得
                html = await page.content()
                hidden = extract_hidden_inputs(html)

                # ✅ フォームにメールを入力して送信
                await page.fill('input[name="email"]', self.email)
                await page.click('button[type="submit"]')
                await asyncio.sleep(2)

                # ✅ 登録成功判定
                content = await page.content()
                if "仮登録メールをお送りしました" not in content:
                    raise RuntimeError("仮登録が失敗した可能性があります")

                logger.info("[LD-GPT-Agent] ✅ 仮登録成功。メール認証を待機します...")

                # ✅ メールから認証リンク取得
                verification_url = None
                async for link in poll_latest_link_gw(self.token, r"https://member\.livedoor\.com/register/.*", timeout=180):
                    verification_url = link
                    break

                if not verification_url:
                    raise RuntimeError("認証リンクの取得に失敗しました")

                logger.info(f"[LD-GPT-Agent] 認証リンクへ移動: {verification_url}")
                await page.goto(verification_url, timeout=30_000)
                await asyncio.sleep(2)

                # ✅ ユーザー情報の入力
                await page.fill('input[name="username"]', self.nickname)
                await page.fill('input[name="password"]', self.password)
                await page.fill('input[name="password2"]', self.password)
                await page.click('button[type="submit"]')
                await asyncio.sleep(3)

                # ✅ APIキー取得ページへ
                await page.goto("https://blog.livedoor.com/settings/api", timeout=30_000)
                html = await page.content()

                # ✅ APIキー抽出
                match = re.search(r'id="api-key">([^<]+)<', html)
                if not match:
                    raise RuntimeError("APIキーの取得に失敗しました")
                api_key = match.group(1).strip()

                # ✅ ブログID（URLから抽出）
                blog_id_match = re.search(r'https://blog\.livedoor\.jp/([a-zA-Z0-9_]+)/', html)
                blog_id = blog_id_match.group(1) if blog_id_match else self.nickname

                logger.info("[LD-GPT-Agent] 🎉 登録完了: blog_id=%s, api_key=%s", blog_id, api_key)

                return {
                    "blog_id": blog_id,
                    "api_key": api_key,
                }

            except Exception as e:
                logger.error(f"[LD-GPT-Agent] エラー: {e}")
                raise

            finally:
                await browser.close()
