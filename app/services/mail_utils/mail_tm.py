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

def create_inbox() -> tuple[str, str]:
    """
    1) 使用可能なドメインを取得 → mail.tm を除外
    2) ランダム email を作成
    3) /accounts で登録
    4) /token で JWT 取得
    Returns
    -------
    (email, jwt)
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
    r.raise_for_status()
    jwt = r.json()["token"]

    return email, jwt


def poll_latest_link_tm(
    jwt: str,
    sender_like: str | None = "@note.com",
    timeout: int = 180,
    interval: int = 6,
) -> Optional[str]:
    """
    受信箱をポーリングし本文内の最初の URL を返す
    """
    # ✅ 修正：requests.Session() を毎回新たに作成＋トークンを強制的に付与
    S2 = requests.Session()
    S2.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {jwt}"  # ✅ トークンをセット
    })

    deadline = time.time() + timeout

    while time.time() < deadline:
        r = S2.get(f"{BASE}/messages")
        _log(r)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error("[mail.tm] AUTH ERROR: %s", e)
            return None  # 🔁 401の時点でリトライせず終了する方が安全

        msgs = sorted(r.json()["hydra:member"], key=lambda x: x["createdAt"], reverse=True)

        for msg in msgs:
            frm = msg.get("from", {}).get("address", "")
            if sender_like and sender_like not in frm:
                continue
            mid = msg["id"]
            body = S2.get(f"{BASE}/messages/{mid}").json()["html"][0]
            links = _links_from_html(body)
            if links:
                return links[0]
        time.sleep(interval)

    logging.error("[mail.tm] verification link not found (timeout)")
    return None

