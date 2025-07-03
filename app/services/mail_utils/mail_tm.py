# -*- coding: utf-8 -*-
"""
Mail.tm で disposable メールを生成し、検証リンクを取得
"""

import random, string, time, requests, logging

API = "https://api.mail.tm"


def _randstr(n=10):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


# ------------------------------------------------------------------
# ① 利用可能ドメインを取ってからアカウントを作成
# ------------------------------------------------------------------
def _pick_domain() -> str:
    """有効なドメインを 1 つ返す（API が複数返す場合あり）"""
    r = requests.get(f"{API}/domains?page=1", timeout=10).json()
    domains = [d["domain"] for d in r["hydra:member"]]
    if not domains:
        raise RuntimeError("mail.tm domains not available")
    return random.choice(domains)


def create_inbox() -> tuple[str, str]:
    """
    新しい INBOX を生成 → (email, JWT token) を返す
    """
    domain = _pick_domain()
    email = f"{_randstr()}@{domain}"
    password = _randstr(12)

    # アカウント作成
    res = requests.post(f"{API}/accounts", json={
        "address": email, "password": password
    }, timeout=10)

    # 既に同じ ID が生成されていた等で 422 が返ったら 1 回だけリトライ
    if res.status_code == 422:
        email = f"{_randstr()}@{domain}"
        res = requests.post(f"{API}/accounts", json={
            "address": email, "password": password
        }, timeout=10)

    res.raise_for_status()

    # トークン取得
    jwt = requests.post(f"{API}/token", json={
        "address": email, "password": password
    }, timeout=10).json()["token"]

    return email, jwt


# ------------------------------------------------------------------
def wait_link(token: str,
              subject_kw: str = "メールアドレスの確認",
              timeout_sec: int = 90) -> str | None:
    """
    検証メールを待ち、本文から note の確認リンクを抽出して返す
    """
    hdrs = {"Authorization": f"Bearer {token}"}
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        msgs = requests.get(f"{API}/messages", headers=hdrs, timeout=10)\
                       .json()["hydra:member"]
        for m in msgs:
            if subject_kw.lower() in m["subject"].lower():
                mid = m["id"]
                html = requests.get(f"{API}/messages/{mid}",
                                    headers=hdrs, timeout=10).json()["html"][0]
                import re
                mt = re.search(r"https://note\.com/[^\"']+signup/[^\"']+", html)
                if mt:
                    return mt.group(0)
        time.sleep(5)

    logging.error("[mail_tm] verification mail not arrived")
    return None
