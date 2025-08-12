# -*- coding: utf-8 -*-
"""
mail.gw API helper
──────────────────────────────
create_inbox()       -> (email, jwt)
poll_latest_link()   -> URL or None
──────────────────────────────
"""
from __future__ import annotations
import os
import secrets
import string
import time
import re
import logging
import httpx
import html
import random
from bs4 import BeautifulSoup
import asyncio
from collections.abc import AsyncGenerator
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)  # ← ✅ logger をモジュールスコープに統一

BASE = "https://api.mail.tm"
USER_AGENT = "Mozilla/5.0 (SEO-Bot)"

def _client():
    return httpx.Client(base_url=BASE,
                        headers={"User-Agent": USER_AGENT},
                        timeout=20)

def _rand_str(n: int = 10) -> str:
    return secrets.token_hex(n // 2)

def _links_from_html(body: str) -> list[str]:
    soup = BeautifulSoup(body, "lxml")
    return [html.unescape(a["href"]) for a in soup.find_all("a", href=True)]

# -------------------- 追加: ドメイン選択と再試行ヘルパ --------------------

def _domain_blacklist_from_env() -> Set[str]:
    v = os.getenv("MAILTM_DOMAIN_BLACKLIST", "")
    return {s.strip().lower() for s in v.split(",") if s.strip()}

def _pick_domain(cli: httpx.Client, blacklist: Optional[Set[str]] = None) -> str:
    """
    mail.tm の /domains から利用可能なドメインを取得し、ブラックリストを除外してランダムに1つ返す
    """
    blacklist = blacklist or set()
    r = cli.get("/domains", timeout=10)
    r.raise_for_status()
    items = r.json().get("hydra:member", [])
    pool = [d.get("domain") for d in items if d.get("domain")]
    # ブラックリスト除外
    pool = [d for d in pool if d.lower() not in blacklist]
    if not pool:
        raise RuntimeError("mail.tm: 利用可能ドメインが空です（ブラックリスト等で除外され過ぎ）")
    return random.choice(pool)

def _create_account_with_retry(cli: httpx.Client, max_attempts: int = 4) -> Tuple[str, str]:
    """
    ランダムドメインでアカウント作成 → 429/409/一部のHTTPエラーはドメイン替えでリトライ。
    戻り値: (email, jwt)
    """
    blacklist = _domain_blacklist_from_env()
    backoff = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            domain = _pick_domain(cli, blacklist)
            username = _rand_str()
            password = _rand_str(12)
            email = f"{username}@{domain}"

            r = cli.post("/accounts", json={"address": email, "password": password}, timeout=20)
            if r.status_code == 201:
                t = cli.post("/token", json={"address": email, "password": password}, timeout=20)
                t.raise_for_status()
                jwt = t.json().get("token")
                if not jwt:
                    raise RuntimeError("mail.tm: token が空です")
                logger.info(f"[mail.tm] ✅ created new inbox: {email}")
                logger.info(f"[mail.tm] ✅ JWT head: {jwt[:10]}...")
                return email, jwt

            if r.status_code in (409, 429):
                logger.warning("[mail.tm] status=%s。ドメイン替えで再試行 (%d/%d)", r.status_code, attempt, max_attempts)
                time.sleep(backoff)
                backoff = min(backoff + 2, 10)
                try:
                    blacklist.add(domain.lower())
                except Exception:
                    pass
                continue

            r.raise_for_status()

        except Exception as e:
            last_exc = e
            logger.warning("[mail.tm] アカウント作成失敗。再試行 (%d/%d): %s", attempt, max_attempts, e)
            time.sleep(backoff)
            backoff = min(backoff + 2, 10)
            continue

    if last_exc:
        logger.error("[mail.tm] アカウント作成に失敗（上限到達）: %s", last_exc)
    return None, None

# --------------------------------------------------------- main API
def create_inbox() -> tuple[str, str]:
    """
    旧実装は 'mail.tm を除外して1発作成' だったが、ランダム＋リトライに変更
    """
    try:
        with _client() as cli:
            email, jwt = _create_account_with_retry(cli, max_attempts=4)
            return email, jwt
    except Exception as e:
        logger.error("[mail.tm] create_inbox 失敗: %s", e)
        return None, None

# --------------------------------------------------------- polling
async def poll_latest_link_gw(
    jwt: str,
    pattern: str = r"https://member\.livedoor\.com/email_auth/commit/[a-zA-Z0-9]+",  # ✅ livedoor 認証リンク
    timeout: int = 240
) -> AsyncGenerator[str, None]:
    logger.info("✅ poll_latest_link_gw が呼び出されました")

    deadline = time.time() + timeout
    headers = {
        "Authorization": f"Bearer {jwt}",
        "User-Agent": USER_AGENT
    }

    try:
        async with httpx.AsyncClient(base_url=BASE, headers=headers, timeout=20) as client:
            poll_count = 0
            sleep_s = 5
            while time.time() < deadline:
                poll_count += 1
                logger.info(f"🔄 ポーリング試行 {poll_count} 回目")

                try:
                    res1 = await client.get("/messages")
                    res1.raise_for_status()
                    messages = res1.json().get("hydra:member", [])

                    logger.info(f"📨 取得メール件数: {len(messages)}")

                    for msg in messages:
                        if msg.get("seen"):
                            continue
                        mid = msg.get("id")
                        if not mid:
                            continue

                        logger.info(f"🆕 新規メールID: {mid} 件名: {msg.get('subject')}")

                        res2 = await client.get(f"/messages/{mid}")
                        res2.raise_for_status()
                        detail = res2.json()
                        html_raw = detail.get("html")

                        html_content = ""
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
                            yield link
                            return

                except Exception as e:
                    logger.warning(f"[mail.gw] メール取得中に例外発生: {e}")

                await asyncio.sleep(sleep_s)
                sleep_s = min(sleep_s + 1, 10)

    except Exception as e:
        logger.error(f"[mail.gw] クライアント接続中に致命的エラー: {e}")

    logger.warning("⏰ poll_latest_link_gw: 認証リンクが見つからないままタイムアウト")
