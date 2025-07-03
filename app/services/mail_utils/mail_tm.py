# -*- coding: utf-8 -*-
"""
Mail.tm で disposable メールを作成 & 受信リンク取得
"""

import random, string, time, requests, logging

API = "https://api.mail.tm"

def _rand_str(n=10):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

# -------------------------------------------------------------

def create_inbox() -> tuple[str, str]:
    """
    新規 INBOX を生成し (address, token) を返す。
    token は後続の GET /messages 用 JWT。
    """
    email = f"{_rand_str()}@mail.tm"
    password = _rand_str(12)

    # 1️⃣ アカウント作成
    resp = requests.post(f"{API}/accounts", json={
        "address": email, "password": password
    }, timeout=10)
    resp.raise_for_status()

    # 2️⃣ トークン取得
    tok = requests.post(f"{API}/token", json={
        "address": email, "password": password
    }, timeout=10).json()["token"]
    return email, tok

# -------------------------------------------------------------

def wait_link(token: str, subject_kw="note", timeout_sec=90) -> str | None:
    """
    subject に `subject_kw` を含むメールを待ち、本文から
    'https://note.com/signup/complete...' を抽出して返す。
    """
    hdrs = {"Authorization": f"Bearer {token}"}
    end = time.time() + timeout_sec
    while time.time() < end:
        msgs = requests.get(f"{API}/messages", headers=hdrs, timeout=10).json()["hydra:member"]
        for m in msgs:
            if subject_kw.lower() in m["subject"].lower():
                # 本文取得
                mid = m["id"]
                full = requests.get(f"{API}/messages/{mid}", headers=hdrs, timeout=10).json()
                html = full["html"][0]  # list[str]
                # リンク抽出
                import re
                m = re.search(r"https://note\.com/[^\"']+signup/[^\"']+", html)
                if m:
                    return m.group(0)
        time.sleep(5)
    logging.error("[mail_tm] verification mail not arrived")
    return None
