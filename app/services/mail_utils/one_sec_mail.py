# -*- coding: utf-8 -*-
"""
1secmail.com API helper
──────────────────────────────
create_inbox()       -> (email, None)   # token は不要
poll_latest_link()   -> URL or None
──────────────────────────────
"""
from __future__ import annotations
import secrets, time, httpx, re, logging

# 末尾スラ無しが公式仕様
BASE = "https://www.1secmail.com/api/v1"

# ---------------------------------------------------------------- utilities
def _rand_name(n: int = 6) -> str:
    return secrets.token_hex(n)

def create_inbox() -> tuple[str, None]:
    """
    使い捨てメールアドレスを生成して返す

    Returns
    -------
    (email, None)   # token は不要
    """
    name   = _rand_name()
    domain = "1secmail.com"             # ほかに .org / .net も可
    return f"{name}@{domain}", None

# ---------------------------------------------------------------- main
def poll_latest_link(
    address: str,
    pattern: str = r"https://member\.livedoor\.com/register/.*",
    timeout: int = 180,
) -> str | None:
    """
    受信箱をポーリングし本文内の最初の URL を返す

    Parameters
    ----------
    address : str          e.g. "abcd@1secmail.com"
    pattern : str          抽出する URL の正規表現
    timeout : int          最大待機秒数

    Returns
    -------
    url or None
    """
    user, dom = address.split("@")
    deadline  = time.time() + timeout
    while time.time() < deadline:
        # ---- getMessages ----
        try:
            r = httpx.get(
                BASE,
                params={"action": "getMessages", "login": user, "domain": dom},
                timeout=20,
            )
            if r.status_code != 200:
                logging.warning("[1secmail] getMessages status=%s", r.status_code)
                time.sleep(5)
                continue
            msgs = r.json()
        except Exception as e:
            logging.warning("[1secmail] getMessages err: %s", e)
            time.sleep(5)
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
                if r.status_code != 200:
                    logging.warning("[1secmail] readMessage status=%s", r.status_code)
                    time.sleep(5)
                    continue
                body = r.json()["body"]
            except Exception as e:
                logging.warning("[1secmail] readMessage err: %s", e)
                time.sleep(5)
                continue

            if (m := re.search(pattern, body)):
                return m.group(0)

        time.sleep(5)

    logging.error("[1secmail] verification link not found (timeout)")
    return None
