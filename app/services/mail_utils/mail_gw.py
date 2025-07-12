# -*- coding: utf-8 -*-
"""
mail.gw API helper
──────────────────────────────
create_inbox()       -> (email, jwt)
poll_latest_link()   -> URL or None
──────────────────────────────
"""
import asyncio 
from __future__ import annotations
import secrets, string, time, re, logging, httpx, html
from bs4 import BeautifulSoup

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
    logging.info("✅ poll_latest_link_gw が呼び出されました")  # ← ログ追加

    stop = time.time() + timeout
    hdr  = {"Authorization": f"Bearer {jwt}", "User-Agent": USER_AGENT}

    # 非同期クライアントで再構成
    async with httpx.AsyncClient(base_url=BASE, headers=hdr, timeout=20) as cli:
        while time.time() < stop:
            res1 = await cli.get("/messages")
            msgs = res1.json()["hydra:member"]

            if msgs:
                mid = msgs[0]["id"]
                res2 = await cli.get(f"/messages/{mid}")
                body = res2.json()["html"][0]

                if (m := re.search(pattern, body)):
                    return m.group(0)

            await asyncio.sleep(5)  # ← 非同期sleep

    logging.error("[mail.gw] verification link not found")
    return None

