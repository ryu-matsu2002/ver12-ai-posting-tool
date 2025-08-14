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
    AtomPubエンドポイントを正規化（末尾スラッシュ除去）。
    """
    ep = (ep or "").strip()
    if not ep:
        # 既定：livedoor公式のAtomPubエンドポイント
        # ※ 以前 blogcms ドメイン直指定にしていたが、endpointカラムを優先して使う
        ep = os.getenv("LIVEDOOR_ATOM_ENDPOINT", "https://livedoor.blogcms.jp/atompub")
    return ep.rstrip("/")


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
        blog_id = getattr(account, "livedoor_blog_id", None)
        if not blog_id:
            raise ValueError("livedoor_blog_id が設定されていません")

        username = getattr(account, "username", None) or blog_id
        api_key = _resolve_api_key(account)
        endpoint = _canon_endpoint(getattr(account, "endpoint", None))

        api_url = f"{endpoint}/{blog_id}/article"  # 末尾スラ無し

        # CDATA 安全化
        safe_body_html = (body_html or "").replace("]]>", "]]]]><![CDATA[>")

        # AtomPub ペイロード
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
            "[LivedoorPost] 投稿開始 blog_id=%s endpoint=%s user=%s key_len=%s",
            blog_id, endpoint, username, len(api_key),
        )

        # タイムアウト付きで軽くリトライ（5xxのみ）
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

        # 失敗時の詳細
        if last_res is not None and last_res.status_code == 401:
            logging.error(
                "[LivedoorPost] 認証失敗(401): endpoint=%s blog_id=%s user=%s body=%s",
                endpoint, blog_id, username, last_res.text[:500]
            )
            return {"ok": False, "error": f"401: Invalid Authenticate (endpoint={endpoint})"}

        status = getattr(last_res, "status_code", "NA")
        body = getattr(last_res, "text", "")[:500]
        logging.error("[LivedoorPost] 投稿失敗 status=%s body=%s", status, body)
        return {"ok": False, "error": f"{status}: {body}"}

    except Exception as e:
        logging.exception("[LivedoorPost] 例外発生")
        return {"ok": False, "error": str(e)}
