# -*- coding: utf-8 -*-
"""
mail.tm API ラッパー（最小実装）
----------------------------------------------
1) create_inbox()       … 使い捨てメールを生成
2) poll_latest_link()   … 新着メールをポーリングして本文から URL を抽出
   * sender_like で差出人フィルタが可能
----------------------------------------------
依存: requests, beautifulsoup4
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE = "https://api.mail.tm"          # ドメインが .tm の場合
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

# ---------------------------------------------------------------------- utils


def _log_resp(resp: requests.Response) -> None:
    logging.debug("[mail.tm] %s %s %s", resp.request.method, resp.url, resp.status_code)


def _parse_links(html_body: str) -> list[str]:
    """本文内の http(s) リンクをすべて返す"""
    soup = BeautifulSoup(html_body, "lxml")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    # HTML エンティティ解除
    return [html.unescape(u) for u in links]


# ---------------------------------------------------------------------- main


def create_inbox() -> tuple[str, str]:
    """
    新規 inbox を作成し (email, JWT) を返す
    Raises : requests.HTTPError
    """
    resp = SESSION.post(f"{BASE}/accounts", json={})
    _log_resp(resp)
    resp.raise_for_status()

    data = resp.json()
    email = data["address"]
    token = data["token"]         # JWT

    SESSION.headers.update({"Authorization": f"Bearer {token}"})
    return email, token


def poll_latest_link(
    jwt: str,
    sender_like: str | None = None,
    timeout: int = 180,
    interval: int = 6,
) -> Optional[str]:
    """
    新着メールを polls して本文中の最初の URL を返す。
    - sender_like: 差出人メールアドレスに含まれる文字列（フィルタ）
    - timeout    : 最大待ち時間 (sec)
    """
    deadline = time.time() + timeout
    SESSION.headers.update({"Authorization": f"Bearer {jwt}"})

    while time.time() < deadline:
        resp = SESSION.get(f"{BASE}/messages")
        _log_resp(resp)
        resp.raise_for_status()
        items = resp.json().get("hydra:member", [])

        # 最新メールから順に
        for msg in sorted(items, key=lambda x: x["createdAt"], reverse=True):
            if sender_like and sender_like not in (msg.get("from", {}).get("address") or ""):
                continue
            # 本文取得
            msg_resp = SESSION.get(f'{BASE}/messages/{msg["id"]}')
            _log_resp(msg_resp)
            msg_resp.raise_for_status()
            body_html = msg_resp.json()["html"][0]
            links = _parse_links(body_html)
            if links:
                return links[0]

        time.sleep(interval)

    logging.error("[mail_tm] verification link not found (timeout)")
    return None
