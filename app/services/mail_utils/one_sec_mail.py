# -*- coding: utf-8 -*-
"""
1secmail.com API helper
──────────────────────────────
create_inbox()       -> (email, None)   # token は不要
poll_latest_link()   -> URL or None
──────────────────────────────
"""
from __future__ import annotations

import logging
import random
import re
import secrets
import time

import httpx

# ------------------------------------------------- constants
BASE = "https://www.1secmail.com/api/v1"       # 末尾スラ無しが公式仕様
DOMAINS = ["1secmail.com", "1secmail.org", "1secmail.net"]

# ------------------------------------------------- utilities
def _rand_name(n: int = 6) -> str:
    return secrets.token_hex(n)

def create_inbox() -> tuple[str, None]:
    """
    使い捨てメールアドレスを生成して返す  
    3 ドメイン (.com / .org / .net) をランダム使用して
    403 ブロックドメインを回避する。
    """
    name   = _rand_name()
    domain = random.choice(DOMAINS)
    return f"{name}@{domain}", None     # token は不要

# ------------------------------------------------- main
def poll_latest_link_sec(
    address: str,
    pattern: str = r"https://member\.livedoor\.com/register/.*",
    timeout: int = 180,
) -> str | None:
    """
    受信箱をポーリングし本文内の最初の URL を返す

    Parameters
    ----------
    address : str  例 "abcd@1secmail.com"
    pattern : str  抽出する URL の正規表現
    timeout : int  最大待機秒数
    """
    user, dom = address.split("@")
    deadline  = time.time() + timeout
    backoff   = 5                       # 初期待機秒

    while time.time() < deadline:
        # ---- getMessages ----
        try:
            r = httpx.get(
                BASE,
                params={
                    "action": "getMessages",
                    "login":  user,
                    "domain": dom,
                },
                timeout=20,
            )
        except Exception as e:
            logging.warning("[1secmail] getMessages req err: %s", e)
            time.sleep(backoff)
            backoff = min(backoff + 5, 60)
            continue

        if r.status_code == 403:               # ← 403 ブロックに遭遇
            logging.warning("[1secmail] 403 blocked, sleep %s sec", backoff)
            time.sleep(backoff)
            backoff = min(backoff + 10, 120)   # バックオフ増加
            continue
        if r.status_code != 200:
            logging.warning("[1secmail] getMessages status=%s", r.status_code)
            time.sleep(backoff)
            backoff = min(backoff + 5, 60)
            continue

        try:
            msgs = r.json()
        except ValueError:
            logging.warning("[1secmail] getMessages invalid JSON: %.80s", r.text)
            time.sleep(backoff)
            backoff = min(backoff + 5, 60)
            continue

        if msgs:
            mid = msgs[0]["id"]

            # ---- readMessage ----
            try:
                r = httpx.get(
                    BASE,
                    params={
                        "action": "readMessage",
                        "login":  user,
                        "domain": dom,
                        "id":     mid,
                    },
                    timeout=20,
                )
            except Exception as e:
                logging.warning("[1secmail] readMessage req err: %s", e)
                time.sleep(backoff)
                backoff = min(backoff + 5, 60)
                continue

            if r.status_code == 403:
                logging.warning("[1secmail] readMessage 403 blocked")
                time.sleep(backoff)
                backoff = min(backoff + 10, 120)
                continue
            if r.status_code != 200:
                logging.warning("[1secmail] readMessage status=%s", r.status_code)
                time.sleep(backoff)
                backoff = min(backoff + 5, 60)
                continue

            try:
                body = r.json()["body"]
            except ValueError:
                logging.warning("[1secmail] readMessage invalid JSON")
                time.sleep(backoff)
                backoff = min(backoff + 5, 60)
                continue

            if (m := re.search(pattern, body)):
                return m.group(0)

        # 何も無ければ待機
        time.sleep(backoff)

    logging.error("[1secmail] verification link not found (timeout)")
    return None
