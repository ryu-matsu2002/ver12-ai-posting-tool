# -*- coding: utf-8 -*-
"""
mail.tm API の極小クライアント
create_inbox()        -> (address, password, token)
wait_verify_link()    -> 検証リンク or None
"""
from __future__ import annotations
import os
import secrets
import string
import time
import re
import httpx
import logging
import random
from typing import Tuple, Set, Optional

BASE = "https://api.mail.tm"
USER_AGENT = "Mozilla/5.0 (SEO-Bot)"

log = logging.getLogger(__name__)

def _client():
    return httpx.Client(base_url=BASE, headers={"User-Agent": USER_AGENT}, timeout=20)

# -------------------- 追加: ドメイン選択と再試行ヘルパ --------------------

def _domain_blacklist_from_env() -> Set[str]:
    v = os.getenv("MAILTM_DOMAIN_BLACKLIST", "")
    return {s.strip().lower() for s in v.split(",") if s.strip()}

def _pick_domain(cli: httpx.Client, blacklist: Optional[Set[str]] = None) -> str:
    """
    mail.tm の /domains から利用可能なドメインを取得し、ブラックリストを除外してランダムに1つ返す
    """
    blacklist = blacklist or set()
    r = cli.get("/domains", timeout=10)
    r.raise_for_status()
    items = r.json().get("hydra:member", [])
    pool = [d.get("domain") for d in items if d.get("domain")]
    # ブラックリスト除外
    pool = [d for d in pool if d.lower() not in blacklist]
    if not pool:
        raise RuntimeError("mail.tm: 利用可能ドメインが空です（ブラックリスト等で除外され過ぎ）")
    return random.choice(pool)

def _create_account_with_retry(cli: httpx.Client, max_attempts: int = 4) -> Tuple[str, str, str]:
    """
    ランダムドメインでアカウント作成 → 429/409/一部のHTTPエラーはドメイン替えでリトライ。
    戻り値: (email, password, token)
    """
    blacklist = _domain_blacklist_from_env()
    backoff = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            domain = _pick_domain(cli, blacklist)
            local  = secrets.token_hex(6)
            email  = f"{local}@{domain}"
            pwd    = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

            r = cli.post("/accounts", json={"address": email, "password": pwd}, timeout=20)
            if r.status_code == 201:
                t = cli.post("/token", json={"address": email, "password": pwd}, timeout=20)
                t.raise_for_status()
                token = t.json().get("token")
                if not token:
                    raise RuntimeError("mail.tm: token が空です")
                log.info("[mail.tm] ✅ created new inbox: %s", email)
                log.info("[mail.tm] ✅ JWT head: %s...", token[:10])
                return email, pwd, token

            # 409(conflict) / 429(rate limit) はドメイン替えで再挑戦
            if r.status_code in (409, 429):
                log.warning("[mail.tm] status=%s。ドメイン替えで再試行 (%d/%d)", r.status_code, attempt, max_attempts)
                time.sleep(backoff)
                backoff = min(backoff + 2, 10)
                # 同じドメインで連発するのを避けるため、一時的にブラックリストに追加（メモリ内）
                try:
                    domain_l = domain.lower()
                    blacklist.add(domain_l)
                except Exception:
                    pass
                continue

            # その他のエラーは例外へ
            r.raise_for_status()

        except Exception as e:
            last_exc = e
            log.warning("[mail.tm] アカウント作成失敗。再試行 (%d/%d): %s", attempt, max_attempts, e)
            time.sleep(backoff)
            backoff = min(backoff + 2, 10)
            continue

    # ここに来たら失敗
    if last_exc:
        raise last_exc
    raise RuntimeError("mail.tm: アカウント作成リトライが上限に達しました")

# -------------------- ここまで追加ヘルパ --------------------

def create_inbox() -> tuple[str, str, str]:
    """
    以前は「最初のドメイン固定」だったのを、ランダム選択＋リトライに変更
    """
    with _client() as cli:
        email, pwd, tok = _create_account_with_retry(cli, max_attempts=4)
        return email, pwd, tok

def _hdr(tok: str):
    return {"Authorization": f"Bearer {tok}", "User-Agent": USER_AGENT}

def wait_verify_link(token: str,
                     pattern=r"https://note\.com/.*verify.*",
                     timeout: int = 240,
                     interval: int = 4) -> str | None:
    """
    受信ポーリング。総待機時間をやや延長し、スリープは最大10秒まで逓増。
    """
    stop = time.time() + timeout
    with _client() as cli:
        while time.time() < stop:
            try:
                msgs = cli.get("/messages", headers=_hdr(token), timeout=20).json().get("hydra:member", [])
                if msgs:
                    # 未読優先で処理（mail.tm は receivedAt 降順でくることが多い）
                    for msg in msgs:
                        mid = msg.get("id")
                        if not mid:
                            continue
                        detail = cli.get(f"/messages/{mid}", headers=_hdr(token), timeout=20).json()
                        html_list = detail.get("html") or []
                        body = html_list[0] if isinstance(html_list, list) and html_list else detail.get("text", "")
                        if not isinstance(body, str):
                            body = ""
                        m = re.search(pattern, body)
                        if m:
                            return m.group(0)
                time.sleep(interval)
                interval = min(interval + 2, 10)
            except Exception as e:
                log.warning("[mail.tm] 受信ポーリング中の例外: %s", e)
                time.sleep(interval)
                interval = min(interval + 2, 10)

    logging.error("[mail.tm] verification link not found (timeout=%ss)", timeout)
    return None

# --- ここから追加（既存互換エイリアス） ----------------------------------------------------

# livedoor_signup 互換エイリアス
def create_disposable_email(seed: str | None = None):
    """
    ライブドア版サインアップが期待する関数名。
    返り値は (email, jwt) のタプル。
    """
    email, _pwd, token = create_inbox()
    return email, token


def poll_inbox(email_or_token: str, pattern: str, timeout: int = 240, interval: int = 6):
    """
    livedoor_signup.py から呼ばれるポーリング関数。
    token を直接受け取り、wait_verify_link() に委譲する。
    """
    # ※ signup 側では token をそのまま渡してくる仕様
    return wait_verify_link(email_or_token, pattern=pattern, timeout=timeout, interval=interval)
# --- 追加ここまで ----------------------------------------------------
