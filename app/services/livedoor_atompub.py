"""
app/services/livedoor_atompub.py
--------------------------------
ライブドアブログ公式 AtomPub API とのやり取りをラップするモジュール。
  * 記事投稿（POST）
  * 記事更新（PUT）
  * 記事削除（DELETE）
  * 画像アップロード用の汎用関数も後で追加可能
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import requests
import xmltodict
from requests.auth import HTTPBasicAuth

# 🔐 既存の共通暗号ユーティリティ
from app.services.blog_signup.crypto_utils import decrypt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# URLユーティリティ
# ---------------------------------------------------------------------
def _entry_base(endpoint: str | None, blog_id: str) -> str:
    """
    endpoint が '.../atompub' でも '.../atompub/<blog_id>' でも
    最終的に '.../atompub/<blog_id>/entry' を返す
    """
    base = (endpoint or "https://livedoor.blogcms.jp/atompub").rstrip("/")
    if not base.endswith(f"/{blog_id}"):
        base = f"{base}/{blog_id}"
    return f"{base}/entry"


def _endpoint(blog_id: str, resource: str = "article") -> str:
    """
    LiveDoor独自のREST風エンドポイントを生成.
    resource:
        - "article"                → 記事コレクション（POST で新規）
        - f"article/{article_id}"  → 特定記事（PUT / DELETE）
    """
    return f"https://livedoor.blogcms.jp/atom/blog/{blog_id}/{resource}"


def _auth(blog_id: str, api_key_enc: str) -> HTTPBasicAuth:
    """暗号化済み AtomPub Key を復号して Basic 認証オブジェクトに変換"""
    api_key = decrypt(api_key_enc)
    return HTTPBasicAuth(blog_id, api_key)


def _build_entry_xml(
    title: str,
    content: str,
    categories: Optional[List[str]] = None,
    draft: bool = False,
) -> str:
    """Atom Entry XML を組み立てて文字列で返す"""
    now = datetime.now(timezone.utc).isoformat()
    entry_dict: Dict[str, Any] = {
        "entry": {
            "@xmlns": "http://purl.org/atom/ns#",
            "title": title,
            "issued": now,
            "modified": now,
            "content": {
                "@type": "text/html",
                "#text": content,
            },
        }
    }

    if categories:
        entry_dict["entry"]["category"] = [
            {"@term": c.strip()} for c in categories if c.strip()
        ]

    if draft:
        # LiveDoor 独自拡張ではなく AtomPub <app:draft>
        entry_dict["entry"]["app:control"] = {
            "@xmlns:app": "http://www.w3.org/2007/app",
            "app:draft": "yes",
        }

    return xmltodict.unparse(entry_dict, pretty=True)


# ---------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------
def post_entry(
    blog_id: str,
    api_key_enc: str,
    title: str,
    content: str,
    categories: Optional[List[str]] = None,
    draft: bool = False,
    timeout: int = 30,
    endpoint: Optional[str] = None,
) -> Tuple[int, str]:
    """
    新規記事を投稿し `(article_id, public_url)` を返す。
    AtomPub形式とREST形式の両方を試す（フォールバックあり）。
    """
    xml_body = _build_entry_xml(title, content, categories, draft)

    primary_url = _entry_base(endpoint, blog_id)   # …/atompub/<id>/entry
    alt_url     = _endpoint(blog_id, "article")    # …/atom/blog/<id>/article

    resp = None
    for i, url in enumerate((primary_url, alt_url)):
        logger.info("[AtomPub] POST %s (try=%d)", url, i+1)
        resp = requests.post(
            url,
            data=xml_body.encode("utf-8"),
            headers={"Content-Type": "application/atom+xml; charset=utf-8"},
            auth=_auth(blog_id, api_key_enc),
            timeout=timeout,
        )
        if resp.status_code in (404, 410) and i == 0:
            logger.warning("[AtomPub] POST %s -> %s, fallback to %s",
                           resp.status_code, url, alt_url)
            continue
        resp.raise_for_status()
        break

    entry = xmltodict.parse(resp.text)["entry"]
    article_id = int(entry["id"].split(".")[-1])
    public_url = next(
        link["@href"]
        for link in entry["link"]
        if link["@rel"] == "alternate" and link["@type"] == "text/html"
    )
    logger.info("[AtomPub] Posted article_id=%s", article_id)
    return article_id, public_url


def update_entry(
    blog_id: str,
    api_key_enc: str,
    article_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    categories: Optional[List[str]] = None,
    timeout: int = 30,
    endpoint: Optional[str] = None,
) -> None:
    """既存記事を更新（全文 PUT）。フォールバックあり。"""
    if not (title or content or categories):
        raise ValueError("update_entry: 変更点がありません")

    primary_base = _entry_base(endpoint, blog_id)
    alt_base     = _endpoint(blog_id, "article")

    # GET
    resp = None
    for i, base in enumerate((primary_base, alt_base)):
        get_url = f"{base}/{article_id}"
        logger.info("[AtomPub] GET %s (try=%d)", get_url, i+1)
        resp = requests.get(get_url, auth=_auth(blog_id, api_key_enc), timeout=timeout)
        if resp.status_code in (404, 410) and i == 0:
            continue
        resp.raise_for_status()
        break

    entry = xmltodict.parse(resp.text)["entry"]

    if title:
        entry["title"] = title
    if content:
        entry["content"]["#text"] = content
    if categories is not None:
        entry["category"] = [{"@term": c.strip()} for c in categories]

    put_xml = xmltodict.unparse({"entry": entry}, pretty=True)

    # PUT
    for i, base in enumerate((primary_base, alt_base)):
        put_url = f"{base}/{article_id}"
        logger.info("[AtomPub] PUT %s (try=%d)", put_url, i+1)
        pr = requests.put(
            put_url,
            data=put_xml.encode("utf-8"),
            headers={"Content-Type": "application/atom+xml; charset=utf-8"},
            auth=_auth(blog_id, api_key_enc),
            timeout=timeout,
        )
        if pr.status_code in (404, 410) and i == 0:
            continue
        pr.raise_for_status()
        break

    logger.info("[AtomPub] Updated article_id=%s", article_id)


def delete_entry(
    blog_id: str,
    api_key_enc: str,
    article_id: int,
    timeout: int = 30,
    endpoint: Optional[str] = None,
) -> None:
    """記事を削除。フォールバックあり。"""
    primary_base = _entry_base(endpoint, blog_id)
    alt_base     = _endpoint(blog_id, "article")

    for i, base in enumerate((primary_base, alt_base)):
        url = f"{base}/{article_id}"
        logger.info("[AtomPub] DELETE %s (try=%d)", url, i+1)
        resp = requests.delete(url, auth=_auth(blog_id, api_key_enc), timeout=timeout)
        if resp.status_code in (404, 410) and i == 0:
            continue
        resp.raise_for_status()
        break

    logger.info("[AtomPub] Deleted article_id=%s", article_id)
