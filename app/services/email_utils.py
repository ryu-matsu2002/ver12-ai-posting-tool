# app/services/email_utils.py
"""
Mail.tm API を用いた『使い捨てメールアドレス自動発行』ユーティリティ
1) create_temp_mail()  : (address, password, token) を返す
2) wait_for_link(token, pattern, timeout) : 最新メール本文から URL を抽出
"""

from __future__ import annotations

import httpx
import time
import logging
import re
import secrets
import string
from typing import Tuple

BASE = "https://api.mail.tm"


# ---------------------------------------------------------------- client factory
def _client() -> httpx.Client:                       # 20 秒に短いタイムアウト
    return httpx.Client(base_url=BASE, timeout=20)


# ---------------------------------------------------------------- create_temp_mail
def create_temp_mail() -> Tuple[str, str, str]:
    """
    Returns
    -------
    (address, password, jwt)
        *429 Too Many Requests* が出た場合は
        2, 4, 8, 16, 32, 64 秒で指数バックオフしながら最大 6 回リトライ。
        それでも成功しなければ RuntimeError を送出する。
    """
    with _client() as cli:

        # 1) 利用可能ドメイン取得
        dom_resp = cli.get("/domains")
        dom_resp.raise_for_status()
        dom = dom_resp.json()["hydra:member"][0]["domain"]

        # 2) アカウント作成 ── 429 時は指数バックオフ
        addr = f"{secrets.token_hex(6)}@{dom}"
        pwd  = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

        for i in range(6):                       # 6 回 → 最大 64 秒待機
            r = cli.post("/accounts", json={"address": addr, "password": pwd})
            if r.status_code != 429:
                r.raise_for_status()
                break
            wait = 2 ** i
            logging.warning("[mail.tm] 429 /accounts → retry in %ss", wait)
            time.sleep(wait)
        else:
            raise RuntimeError("mail.tm 429 on /accounts (too many retries)")

        # 3) JWT トークン取得 ── 同じくリトライ
        for i in range(6):
            r = cli.post("/token", json={"address": addr, "password": pwd})
            if r.status_code != 429:
                r.raise_for_status()
                tok = r.json()["token"]
                break
            wait = 2 ** i
            logging.warning("[mail.tm] 429 /token → retry in %ss", wait)
            time.sleep(wait)
        else:
            raise RuntimeError("mail.tm 429 on /token (too many retries)")

        return addr, pwd, tok


# ---------------------------------------------------------------- wait_for_link
def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def wait_for_link(
    token: str,
    pattern: str = r"https?://[^\s\"'<>]+",
    timeout: int = 60,
) -> str | None:
    """
    指定秒以内に受信した最初のメール本文から URL を抽出して返す。

    Parameters
    ----------
    token   : JWT returned by create_temp_mail()
    pattern : 正規表現で表した抽出対象 URL
    timeout : 秒

    Returns
    -------
    str | None
        最初に見つけた URL。タイムアウトした場合は None。
    """
    deadline = time.time() + timeout
    with _client() as cli:
        while time.time() < deadline:
            msgs = cli.get("/messages", headers=_auth_headers(token)).json()["hydra:member"]
            if msgs:
                mid  = msgs[0]["id"]
                body = cli.get(f"/messages/{mid}", headers=_auth_headers(token)).json()["html"]
                if body and (m := re.search(pattern, body)):
                    return m.group(0)
            time.sleep(3)

    logging.warning("[email_utils] メール受信待ちタイムアウト")
    return None
