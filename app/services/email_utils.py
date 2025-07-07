# app/services/mail_utils/mail_tm.py
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

BASE = "https://api.mail.tm"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})


# ---------------------------------------------------------------- utilities
def _rand_str(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _links_from_html(body: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "lxml")
    return [html.unescape(a["href"]) for a in soup.find_all("a", href=True)]


def _log(resp: requests.Response) -> None:
    logging.debug("[mail.tm] %s %s → %s", resp.request.method, resp.url, resp.status_code)


# ---------------------------------------------------------------- main API with 429-retry
def _post_with_retry(url: str, json_data: dict, max_retry: int = 6) -> requests.Response:
    """
    429 が返ったら指数バックオフで再試行
    2, 4, 8, 16, 32, 64 秒 → 最大 ~2 分
    """
    for i in range(max_retry):
        r = S.post(url, json=json_data)
        _log(r)
        if r.status_code != 429:
            return r
        wait = 2 ** i
        logging.warning("[mail.tm] 429 %s retry in %ss", url.rsplit('/', 1)[-1], wait)
        time.sleep(wait)
    raise RuntimeError(f"mail.tm 429 on {url} (too many retries)")


def create_inbox() -> tuple[str, str]:
    """
    1) ドメイン取得 → ランダム email 作成
    2) /accounts で登録（429 時はリトライ）
    3) /token で JWT 取得      （429 時はリトライ）

    Returns
    -------
    (email, jwt)
    """
    # 1️⃣ ドメイン
    r = S.get(f"{BASE}/domains")
    _log(r)
    r.raise_for_status()
    domain = r.json()["hydra:member"][0]["domain"]

    email = f"{_rand_str()}@{domain}"
    pwd = _rand_str(12)  # 使い捨てパス

    # 2️⃣ アカウント作成（429 対応）
    r = _post_with_retry(f"{BASE}/accounts", {"address": email, "password": pwd})
    r.raise_for_status()

    # 3️⃣ トークン取得（429 対応）
    r = _post_with_retry(f"{BASE}/token", {"address": email, "password": pwd})
    r.raise_for_status()
    jwt = r.json()["token"]

    S.headers.update({"Authorization": f"Bearer {jwt}"})
    return email, jwt


def poll_latest_link(
    jwt: str,
    sender_like: str | None = "@livedoor",
    timeout: int = 180,
    interval: int = 6,
) -> Optional[str]:
    """
    受信箱をポーリングし本文内の最初の URL を返す
    """
    deadline = time.time() + timeout
    S.headers.update({"Authorization": f"Bearer {jwt}"})

    while time.time() < deadline:
        r = S.get(f"{BASE}/messages")
        _log(r)
        r.raise_for_status()
        msgs = sorted(r.json()["hydra:member"], key=lambda x: x["createdAt"], reverse=True)

        for msg in msgs:
            frm = msg.get("from", {}).get("address", "")
            if sender_like and sender_like not in frm:
                continue
            mid = msg["id"]
            body = S.get(f"{BASE}/messages/{mid}").json()["html"][0]
            links = _links_from_html(body)
            if links:
                return links[0]
        time.sleep(interval)

    logging.error("[mail.tm] verification link not found (timeout)")
    return None
