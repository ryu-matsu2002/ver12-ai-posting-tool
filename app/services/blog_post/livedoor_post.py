# app/services/blog_post/livedoor_post.py
import os
import re
import logging
from datetime import datetime
from xml.sax.saxutils import escape

import requests
from requests.auth import HTTPBasicAuth

from app.services.blog_signup.crypto_utils import decrypt  # 復号はフォールバックでのみ使用


def _canon_endpoint(ep: str | None) -> str:
    """
    AtomPubエンドポイントを正規化（常に .../atompub に揃える）。
    endpoint に /<blog_id> や /article が付いて保存されていても削ぎ落とす。
    """
    ep = (ep or "").strip()
    if not ep:
        ep = os.getenv("LIVEDOOR_ATOM_ENDPOINT", "https://livedoor.blogcms.jp/atompub")

    ep = ep.rstrip("/")

    # livedoor.blogcms.jp の /atompub 配下に余計なパスが付いていたら /atompub に戻す
    m = re.match(r"^(https://livedoor\.blogcms\.jp/atompub)(?:/.*)?$", ep)
    if m:
        return m.group(1)

    # それ以外が来ても保険でベースに固定
    return "https://livedoor.blogcms.jp/atompub"

def _norm_blog_id(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s



def _resolve_api_key(account) -> str:
    """
    APIキーの決定ロジック：
    1) account.api_key（平文）を最優先
    2) ない場合は atompub_key_enc を復号（失敗は例外）
    """
    if getattr(account, "api_key", None):
        return account.api_key.strip()

    enc = getattr(account, "atompub_key_enc", None)
    if not enc:
        raise ValueError("APIキーが見つかりません（api_key / atompub_key_enc いずれも未設定）")

    # 復号は失敗を黙殺せず、例外にする（平文扱いの危険なフォールバックはしない）
    key = decrypt(enc)
    if not key:
        raise ValueError("APIキーの復号に失敗しました（atompub_key_enc が不正）")
    return key.strip()


def extract_post_url(xml_response: str) -> str | None:
    """
    Livedoor AtomPub レスポンスから公開URLを抽出
    """
    m = re.search(
        r"<link\s+rel=['\"]alternate['\"]\s+type=['\"]text/html['\"]\s+href=['\"](.*?)['\"]",
        xml_response,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1) if m else None


def post_livedoor_article(account, title: str, body_html: str):
    """
    ライブドアブログに記事を投稿する。
    account: ExternalBlogAccount
    """
    try:
        raw_blog_id = getattr(account, "livedoor_blog_id", None)
        if not raw_blog_id:
            raise ValueError("livedoor_blog_id が設定されていません")

        blog_id = _norm_blog_id(raw_blog_id)

        username = blog_id
        api_key = _resolve_api_key(account)
        endpoint = _canon_endpoint(getattr(account, "atompub_endpoint", None))

        # ✅ 最終URLは必ず .../atompub/<blog_id>/article の形になる
        api_url = f"{endpoint}/{blog_id}/article"

        # CDATA 安全化
        safe_body_html = (body_html or "").replace("]]>", "]]]]><![CDATA[>")

        xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom">
  <title>{escape(title or "")}</title>
  <content type="text/html"><![CDATA[{safe_body_html}]]></content>
</entry>
"""

        headers = {
            "Content-Type": "application/atom+xml;type=entry;charset=utf-8",
        }
        auth = HTTPBasicAuth(username, api_key)

        logging.info(
            "[LivedoorPost] 投稿開始 blog_id=%s endpoint=%s url=%s user=%s key_len=%s",
            blog_id, endpoint, api_url, username, len(api_key),
        )

        last_res = None
        for attempt in range(2):
            res = requests.post(
                api_url, data=xml_payload.encode("utf-8"),
                headers=headers, auth=auth, timeout=30
            )
            last_res = res

            if res.status_code in (200, 201):
                post_url = extract_post_url(res.text) or f"https://{blog_id}.livedoor.blog/"
                logging.info("[LivedoorPost] 投稿成功: %s", post_url)
                return {"ok": True, "url": post_url, "posted_at": datetime.utcnow()}

            if 500 <= res.status_code < 600 and attempt == 0:
                logging.warning("[LivedoorPost] 5xx再試行 status=%s body=%s", res.status_code, res.text[:400])
                continue
            break

        if last_res is not None and last_res.status_code == 401:
            logging.error(
                "[LivedoorPost] 認証失敗(401): endpoint=%s url=%s blog_id=%s user=%s body=%s",
                endpoint, api_url, blog_id, username, last_res.text[:500]
            )
            return {"ok": False, "error": f"401: Invalid Authenticate (endpoint={endpoint})"}

        status = getattr(last_res, "status_code", "NA")
        body = getattr(last_res, "text", "")[:500]
        logging.error("[LivedoorPost] 投稿失敗 status=%s url=%s body=%s", status, api_url, body)
        return {"ok": False, "error": f"{status}: {body}"}

    except Exception as e:
        logging.exception("[LivedoorPost] 例外発生")
        return {"ok": False, "error": str(e)}
