# -*- coding: utf-8 -*-
"""
mail.tm API の極小クライアント
create_inbox()        -> (address, password, token)
wait_verify_link()    -> 検証リンク or None
"""
from __future__ import annotations
import secrets, string, time, re, httpx, logging

BASE = "https://api.mail.tm"
USER_AGENT = "Mozilla/5.0 (SEO-Bot)"

def _client():
    return httpx.Client(base_url=BASE, headers={"User-Agent": USER_AGENT}, timeout=20)

def create_inbox() -> tuple[str, str, str]:
    with _client() as cli:
        dom  = cli.get("/domains").json()["hydra:member"][0]["domain"]
        addr = f"{secrets.token_hex(6)}@{dom}"
        pwd  = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        cli.post("/accounts", json={"address": addr, "password": pwd}).raise_for_status()
        tok = cli.post("/token",    json={"address": addr, "password": pwd}).json()["token"]
        return addr, pwd, tok

def _hdr(tok: str):
    return {"Authorization": f"Bearer {tok}", "User-Agent": USER_AGENT}

def wait_verify_link(token: str,
                     pattern=r"https://note\.com/.*verify.*",
                     timeout: int = 180) -> str | None:
    stop = time.time() + timeout
    with _client() as cli:
        while time.time() < stop:
            msgs = cli.get("/messages", headers=_hdr(token)).json()["hydra:member"]
            if msgs:
                body = cli.get(f"/messages/{msgs[0]['id']}",
                               headers=_hdr(token)).json()["html"][0]
                if (m := re.search(pattern, body)):
                    return m.group(0)
            time.sleep(5)
    logging.error("[mail.tm] verification link not found")
    return None
