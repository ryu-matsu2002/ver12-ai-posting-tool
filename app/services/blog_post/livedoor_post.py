# app/services/blog_post/livedoor_post.py
import requests
from requests.auth import HTTPBasicAuth
import re
import logging
from app.services.blog_signup.crypto_utils import decrypt
from datetime import datetime
from xml.sax.saxutils import escape


def post_livedoor_article(account, title, body_html):
    """
    ライブドアブログに記事を投稿する
    account: ExternalBlogAccount インスタンス（APIキーなどを保持）
    """
    try:
        # livedoor_blog_id を使う
        if not account.livedoor_blog_id:
            raise ValueError("livedoor_blog_id が設定されていません")

        blog_id = account.livedoor_blog_id
        # APIエンドポイント（末尾の / を付ける）
        api_url = f"https://livedoor.blogcms.jp/atompub/{blog_id}/article/"

        # CDATA安全化（]]> が本文中にある場合の対策）
        safe_body_html = body_html.replace("]]>", "]]]]><![CDATA[>")

        # AtomPub用XMLペイロード
        xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
        <entry xmlns="http://www.w3.org/2005/Atom">
            <title>{escape(title)}</title>
            <content type="text/html"><![CDATA[{safe_body_html}]]></content>
        </entry>
        """

        # APIキーを復号
        api_key_dec = decrypt(account.atompub_key_enc)

        auth = HTTPBasicAuth(blog_id, api_key_dec)
        headers = {"Content-Type": "application/atom+xml; charset=utf-8"}

        logging.info(f"[LivedoorPost] 投稿開始: blog_id={blog_id}, url={api_url}")

        res = requests.post(api_url, data=xml_payload.encode("utf-8"), headers=headers, auth=auth)

        if res.status_code in (200, 201):
            post_url = extract_post_url(res.text)
            # フォールバックURL（URL取得できなかった場合）
            if not post_url:
                post_url = f"https://{blog_id}.livedoor.blog/"
                logging.warning("[LivedoorPost] 投稿成功したがURL抽出に失敗。フォールバックURLを使用します。")
            logging.info(f"[LivedoorPost] 投稿成功: {post_url}")
            return {"ok": True, "url": post_url, "posted_at": datetime.utcnow()}
        else:
            logging.error(f"[LivedoorPost] 投稿失敗: status={res.status_code}, response={res.text}")
            return {"ok": False, "error": f"{res.status_code}: {res.text}"}

    except Exception as e:
        logging.exception("[LivedoorPost] 例外発生")
        return {"ok": False, "error": str(e)}


def extract_post_url(xml_response):
    """
    Livedoor AtomPubのレスポンスXMLから投稿URLを抽出
    """
    match = re.search(r"<link\s+rel=['\"]alternate['\"]\s+type=['\"]text/html['\"]\s+href=['\"](.*?)['\"]", xml_response)
    return match.group(1) if match else None
