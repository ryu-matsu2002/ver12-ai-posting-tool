# -*- coding: utf-8 -*-
"""
1secmail.com API helper
──────────────────────────────
create_inbox()       -> (email, None)   # token は不要
poll_latest_link()   -> URL or None
──────────────────────────────
"""
import secrets, time, httpx, re, logging

BASE = "https://www.1secmail.com/api/v1/"

def _rand_name() -> str:
    return secrets.token_hex(6)

# ---------------------------------------------------------------- utilities
def create_inbox() -> tuple[str, None]:
    """
    使い捨てメールアドレスを生成して返す
    Returns
    -------
    (email, None)
    """
    name   = _rand_name()
    domain = "1secmail.com"             # 他に .org / .net も利用可能
    return f"{name}@{domain}", None

def poll_latest_link(
    address: str,
    pattern: str = r"https://member\.livedoor\.com/register/.*",
    timeout: int = 180,
) -> str | None:
    """
    受信箱をポーリングし本文内の最初の URL を返す
    """
    user, dom = address.split("@")
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(BASE, params={
            "action": "getMessages",
            "login":  user,
            "domain": dom,
        }, timeout=20)
        msgs = r.json()
        if msgs:
            mid  = msgs[0]["id"]
            body = httpx.get(BASE, params={
                "action": "readMessage",
                "login":  user,
                "domain": dom,
                "id":     mid,
            }, timeout=20).json()["body"]
            if (m := re.search(pattern, body)):
                return m.group(0)
        time.sleep(5)
    logging.error("[1secmail] verification link not found (timeout)")
    return None
