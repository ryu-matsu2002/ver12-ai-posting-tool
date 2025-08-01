# -*- coding: utf-8 -*-
"""
mail.tm API helper
──────────────────────────────
create_inbox()       → (email, jwt)
poll_latest_link()   → 最初に見つけた URL を返す
──────────────────────────────
"""

from __future__ import annotations

import html
import logging
import random
import re
import string
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE = "https://api.mail.tm"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})

# ---------------------------------------------------------------- utilities

def _rand_str(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _links_from_html(body: str) -> list[str]:
    soup = BeautifulSoup(body, "lxml")
    return [html.unescape(a["href"]) for a in soup.find_all("a", href=True)]

def _log(resp: requests.Response) -> None:
    logging.debug("[mail.tm] %s %s → %s", resp.request.method, resp.url, resp.status_code)

# ---------------------------------------------------------------- main API

def create_inbox() -> tuple[str, str, str]:
    import logging
    logger = logging.getLogger(__name__)
    """
    1) 使用可能なドメインを取得 → mail.tm を除外
    2) ランダム email を作成
    3) /accounts で登録
    4) /token で JWT 取得
    Returns
    -------
     (email, pwd, jwt)
    """
    # ドメイン一覧を取得
    r = S.get(f"{BASE}/domains")
    _log(r)
    r.raise_for_status()
    domains = [d["domain"] for d in r.json()["hydra:member"]]

    # mail.tm を除外して使用（fallbackあり）
    usable_domains = [d for d in domains if "mail.tm" not in d]
    domain = random.choice(usable_domains or domains)  # 使えるのがなければ fallback

    email = f"{_rand_str()}@{domain}"
    pwd = _rand_str(12)

    # アカウント作成
    r = S.post(f"{BASE}/accounts", json={"address": email, "password": pwd})
    _log(r)
    r.raise_for_status()

    # トークン取得
    r = S.post(f"{BASE}/token", json={"address": email, "password": pwd})
    _log(r)
    try:
        r.raise_for_status()
        jwt = r.json().get("token")
        if not jwt:
            logger.error("[mail.tm] JWTが取得できませんでした（トークン=None）: %s", r.text)
            return None, None
    except Exception as e:
        logger.exception("[mail.tm] JWT取得中に例外が発生しました")
        return None, None
    
    logger.info(f"[mail.tm] ✅ created new inbox: {email}")
    logger.info(f"[mail.tm] ✅ password: {pwd}")
    logger.info(f"[mail.tm] ✅ JWT head: {jwt[:10]}...")

    return email, pwd, jwt



import asyncio
import logging
from typing import Optional
import httpx  # 非同期クライアント
from .html_utils import _links_from_html  # ※既存のリンク抽出関数
# BASE = "https://api.mail.tm" は既存と同じ前提


BASE = "https://api.mail.tm"

async def poll_latest_link_tm_async(
    jwt: str,
    sender_like: str | None = "@livedoor",
    timeout: int = 180,
    interval: int = 6,
) -> Optional[str]:
    """
    非同期で受信箱をポーリングして livedoor 認証リンクを抽出
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {jwt}",
    }

    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(f"{BASE}/messages")
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logging.error("[mail.tm] AUTH ERROR: %s", e)
                break
            except Exception as e:
                logging.warning("[mail.tm] unexpected error: %s", e)
                await asyncio.sleep(interval)
                continue

            msgs = sorted(r.json().get("hydra:member", []), key=lambda x: x["createdAt"], reverse=True)

            for msg in msgs:
                subject = msg.get("subject") or ""  # None なら空文字に
                sender = msg.get("from", {}).get("address", "（送信者不明）")

                # 🔍 ステップ①: すべての件名と送信者を表示（subjectがNoneでも空文字になるので安全）
                print(f"📩 件名: {subject} ｜ 送信者: {sender}")

                # 🔍 ステップ②: livedoorのフィルターを弱めたい場合は以下を一時的にコメントアウト
                # if "livedoor" not in subject.lower():
                #     continue

                # 🔍 ステップ③: 差出人のメールアドレスでのフィルタ（必要であれば残す）
                frm = msg.get("from", {}).get("address", "")
                if sender_like and sender_like not in frm:
                    continue

                mid = msg["id"]
                try:
                    body_resp = await client.get(f"{BASE}/messages/{mid}")
                    body_resp.raise_for_status()
                    body_html_list = body_resp.json().get("html", [])
                    if not body_html_list:
                        continue
                    body = body_html_list[0]
                    links = _links_from_html(body)

                    # livedoor 認証リンクに絞る
                    livedoor_links = [
                        link for link in links if "email_auth/commit" in link
                    ]
                    if livedoor_links:
                        logging.info(f"[mail.tm] ✅ 認証リンク検出: {livedoor_links[0]}")
                        return livedoor_links[0]

                except Exception as e:
                    logging.warning("[mail.tm] failed to parse message %s: %s", mid, e)
                    continue


            await asyncio.sleep(interval)

    logging.error("[mail.tm] ❌ livedoor認証リンクが見つかりませんでした（timeout）")
    return None
