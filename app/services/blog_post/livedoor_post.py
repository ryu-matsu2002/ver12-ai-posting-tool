# app/services/blog_post/livedoor_post.py
import os
import re
import logging
from datetime import datetime
from xml.sax.saxutils import escape

import requests
from requests.auth import HTTPBasicAuth

from app.services.blog_signup.crypto_utils import decrypt  # 復号はフォールバックでのみ使用


# --- URL 正規化ユーティリティ -------------------------------------------------

def _canon_endpoint(ep: str | None) -> str:
    """
    AtomPubエンドポイントを正規化（末尾スラッシュ除去）。
    空なら既定値を返す。
    """
    ep = (ep or "").strip()
    if not ep:
        ep = os.getenv("LIVEDOOR_ATOM_ENDPOINT", "https://livedoor.blogcms.jp/atompub")
    return ep.rstrip("/")


def _entry_base(endpoint: str | None, blog_id: str) -> str:
    """
    endpoint が '.../atompub' でも '.../atompub/<blog_id>' でも
    最終的に '.../atompub/<blog_id>/entry' を返す。
    """
    base = _canon_endpoint(endpoint)
    if not base.endswith(f"/{blog_id}"):
        base = f"{base}/{blog_id}"
    return f"{base}/entry"


def _legacy_article_collection(blog_id: str) -> str:
    """
    旧パス（404/410時のフォールバック用）:
    https://livedoor.blogcms.jp/atom/blog/<blog_id>/article
    """
    return f"https://livedoor.blogcms.jp/atom/blog/{blog_id}/article"


# --- 認証/キー ---------------------------------------------------------------

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

    key = decrypt(enc)
    if not key:
        raise ValueError("APIキーの復号に失敗しました（atompub_key_enc が不正）")
    return key.strip()


# --- レスポンス解析 -----------------------------------------------------------

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


# --- メイン -------------------------------------------------------------------

def post_livedoor_article(account, title: str, body_html: str):
    """
    ライブドアブログに記事を投稿する。
    account: ExternalBlogAccount
    """
    try:
        blog_id = getattr(account, "livedoor_blog_id", None)
        if not blog_id:
            raise ValueError("livedoor_blog_id が設定されていません")

        # Basic 認証のユーザー名は blog_id を使用（既存互換で username があれば優先）
        username = getattr(account, "username", None) or blog_id
        api_key = _resolve_api_key(account)

        # endpoint 正規化 & 正しい投稿先生成
        endpoint = _canon_endpoint(getattr(account, "atompub_endpoint", None))
        primary_url = _entry_base(endpoint, blog_id)              # …/atompub/<id>/entry
        fallback_url = _legacy_article_collection(blog_id)        # …/atom/blog/<id>/article

        # CDATA 安全化
        safe_body_html = (body_html or "").replace("]]>", "]]]]><![CDATA[>")

        # AtomPub ペイロード（2005 Atom）
        xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom">
  <title>{escape(title or "")}</title>
  <content type="text/html"><![CDATA[{safe_body_html}]]></content>
</entry>
"""

        headers = {
            # Livedoorは ';type=entry' なしでも受け付ける。統一して charset のみにする
            "Content-Type": "application/atom+xml; charset=utf-8",
        }
        auth = HTTPBasicAuth(username, api_key)

        logging.info(
            "[LivedoorPost] 投稿開始 blog_id=%s endpoint=%s user=%s key_len=%s",
            blog_id, endpoint, username, len(api_key),
        )

        # --- 1st: 正式パス（…/atompub/<id>/entry） --------------------------------
        logging.info("[LivedoorPost] POST %s", primary_url)
        res = requests.post(primary_url, data=xml_payload.encode("utf-8"),
                            headers=headers, auth=auth, timeout=30)

        # 5xx は一回だけ再試行
        if 500 <= res.status_code < 600:
            logging.warning("[LivedoorPost] 5xx再試行 status=%s body=%s",
                            res.status_code, res.text[:400])
            res = requests.post(primary_url, data=xml_payload.encode("utf-8"),
                                headers=headers, auth=auth, timeout=30)

        if res.status_code in (200, 201):
            post_url = extract_post_url(res.text) or f"https://{blog_id}.livedoor.blog/"
            logging.info("[LivedoorPost] 投稿成功: %s", post_url)
            return {"ok": True, "url": post_url, "posted_at": datetime.utcnow()}

        # 401 は認証エラーで即終了
        if res.status_code == 401:
            logging.error(
                "[LivedoorPost] 認証失敗(401): url=%s blog_id=%s user=%s body=%s",
                primary_url, blog_id, username, res.text[:500]
            )
            return {"ok": False, "error": f"401: Invalid Authenticate (url={primary_url})"}

        # 404/410 は旧パスへフォールバック
        if res.status_code in (404, 410):
            logging.warning("[LivedoorPost] 404/410 フォールバック開始 -> %s", fallback_url)
            res2 = requests.post(fallback_url, data=xml_payload.encode("utf-8"),
                                 headers=headers, auth=auth, timeout=30)

            if 500 <= res2.status_code < 600:
                logging.warning("[LivedoorPost] 5xx再試行(フォールバック) status=%s body=%s",
                                res2.status_code, res2.text[:400])
                res2 = requests.post(fallback_url, data=xml_payload.encode("utf-8"),
                                     headers=headers, auth=auth, timeout=30)

            if res2.status_code in (200, 201):
                post_url = extract_post_url(res2.text) or f"https://{blog_id}.livedoor.blog/"
                logging.info("[LivedoorPost] 投稿成功(フォールバック): %s", post_url)
                return {"ok": True, "url": post_url, "posted_at": datetime.utcnow()}

            # フォールバックも失敗
            logging.error(
                "[LivedoorPost] フォールバック失敗 status=%s body=%s",
                res2.status_code, res2.text[:500]
            )
            return {"ok": False, "error": f"{res2.status_code}: {res2.text[:500]}"}

        # その他の失敗
        logging.error("[LivedoorPost] 投稿失敗 status=%s body=%s", res.status_code, res.text[:500])
        return {"ok": False, "error": f"{res.status_code}: {res.text[:500]}"}

    except Exception as e:
        logging.exception("[LivedoorPost] 例外発生")
        return {"ok": False, "error": str(e)}
