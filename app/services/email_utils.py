# app/services/email_utils.py
"""
Mail.tm API を用いた『使い捨てメールアドレス自動発行』ユーティリティ
1) create_temp_mail()  : (address, password, token) を返す
2) wait_for_link(token, pattern, timeout) : 最新メール本文から URL を抽出
"""

import httpx, time, logging, re
from typing import Tuple

BASE = "https://api.mail.tm"

def _client():
    return httpx.Client(base_url=BASE, timeout=20)

def create_temp_mail() -> Tuple[str, str, str]:
    """
    Returns:
        address  … 新規メールアドレス
        password … ランダムに生成されたパスワード
        token    … 認証用 JWT (後続の GET /messages で使用)
    """
    with _client() as cli:
        # 1) 利用可能ドメイン取得
        dom = cli.get("/domains").json()["hydra:member"][0]["domain"]

        # 2) アカウント作成
        import secrets, string
        addr = f"{secrets.token_hex(6)}@{dom}"
        pwd  = "".join(secrets.choice(string.ascii_letters+string.digits) for _ in range(12))
        r = cli.post("/accounts", json={"address": addr, "password": pwd})
        r.raise_for_status()

        # 3) トークン取得
        tok = cli.post("/token", json={"address": addr, "password": pwd}).json()["token"]
        return addr, pwd, tok

def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}

def wait_for_link(token: str,
                  pattern: str = r"https?://[^\s\"'<>]+",
                  timeout: int = 60) -> str | None:
    """
    指定秒以内に受信した最初のメール本文から URL を抽出して返す。

    Args:
        token   : create_temp_mail() で得た JWT
        pattern : 抽出したい URL を表す正規表現
        timeout : 秒

    Returns:
        str | None … マッチする最初の URL
    """
    stop = time.time() + timeout
    with _client() as cli:
        while time.time() < stop:
            msgs = cli.get("/messages", headers=_auth_headers(token)).json()["hydra:member"]
            if msgs:
                msg_id = msgs[0]["id"]
                body   = cli.get(f"/messages/{msg_id}", headers=_auth_headers(token)).json()["html"]
                if body and (m := re.search(pattern, body)):
                    return m.group(0)
            time.sleep(3)
    logging.warning("[email_utils] メール受信待ちタイムアウト")
    return None
