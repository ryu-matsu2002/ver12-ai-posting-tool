# -*- coding: utf-8 -*-
"""
mail.gw API helper
──────────────────────────────
create_inbox()       -> (email, jwt)
poll_latest_link()   -> URL or None
──────────────────────────────
"""

from __future__ import annotations
import secrets, string, time, re, logging, httpx, html
from bs4 import BeautifulSoup
import asyncio 

BASE = "https://api.mail.gw"
USER_AGENT = "Mozilla/5.0 (SEO-Bot)"

def _client():
    return httpx.Client(base_url=BASE,
                        headers={"User-Agent": USER_AGENT},
                        timeout=20)

def _rand_str(n: int = 10) -> str:
    return secrets.token_hex(n//2)

def _links_from_html(body: str) -> list[str]:
    soup = BeautifulSoup(body, "lxml")
    return [html.unescape(a["href"]) for a in soup.find_all("a", href=True)]

# --------------------------------------------------------- main API
def create_inbox() -> tuple[str, str]:
    with _client() as cli:
        dom  = cli.get("/domains").json()["hydra:member"][0]["domain"]
        addr = f"{_rand_str()}@{dom}"
        pwd  = ''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(12))
        cli.post("/accounts", json={"address": addr, "password": pwd}).raise_for_status()
        jwt = cli.post("/token", json={"address": addr, "password": pwd}).json()["token"]
        return addr, jwt       # jwt は Bearer 認証で使用


# 非同期関数に変更
async def poll_latest_link_gw(
    jwt: str,
    pattern: str = r"https://member\.livedoor\.com/register/.*",
    timeout: int = 180
) -> str | None:
    """
    メールボックスを定期的にポーリングし、指定のリンクパターンが含まれる認証リンクを探す。
    """
    logger = logging.getLogger(__name__)
    logger.info("✅ poll_latest_link_gw が呼び出されました")

    deadline = time.time() + timeout
    headers = {
        "Authorization": f"Bearer {jwt}",
        "User-Agent": USER_AGENT
    }

    try:
        async with httpx.AsyncClient(base_url=BASE, headers=headers, timeout=20) as client:
            while time.time() < deadline:
                try:
                    res1 = await client.get("/messages")
                    res1.raise_for_status()
                    data = res1.json()
                    messages = data.get("hydra:member", [])

                    for msg in messages:
                        if msg.get("seen"):
                            continue
                        mid = msg.get("id")
                        if not mid:
                            continue

                        res2 = await client.get(f"/messages/{mid}")
                        res2.raise_for_status()
                        detail = res2.json()

                        html_raw = detail.get("html")
                        if isinstance(html_raw, list):
                            html_content = html_raw[0] if html_raw else ""
                        elif isinstance(html_raw, str):
                            html_content = html_raw
                        else:
                            logger.warning("⚠️ html フィールドが不正な形式: %s", type(html_raw))
                            continue

                        match = re.search(pattern, html_content)
                        if match:
                            link = match.group(0)
                            logger.info("✅ 認証リンクを検出: %s", link)
                            return link

                except Exception as e:
                    logger.warning(f"[mail.gw] メール取得中に例外発生: {e}")

                await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"[mail.gw] クライアント接続中に致命的エラー: {e}")

    logger.warning("⏰ poll_latest_link_gw: 認証リンクが見つからないままタイムアウト")
    return None
