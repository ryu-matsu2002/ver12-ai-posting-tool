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
        api_url = f"https://livedoor.blogcms.jp/atompub/{blog_id}/article"

        xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
        <entry xmlns="http://www.w3.org/2005/Atom">
            <title>{escape(title)}</title>
            <content type="text/html"><![CDATA[{body_html}]]></content>
        </entry>
        """

        api_key_dec = decrypt(account.atompub_key_enc)

        auth = HTTPBasicAuth(blog_id, api_key_dec)
        headers = {"Content-Type": "application/atom+xml; charset=utf-8"}

        res = requests.post(api_url, data=xml_payload.encode("utf-8"), headers=headers, auth=auth)

        if res.status_code in (200, 201):
            post_url = extract_post_url(res.text)
            logging.info(f"[LivedoorPost] 投稿成功: {post_url}")
            return {"ok": True, "url": post_url, "posted_at": datetime.utcnow()}
        else:
            logging.error(f"[LivedoorPost] 投稿失敗: {res.status_code} {res.text}")
            return {"ok": False, "error": f"{res.status_code}: {res.text}"}

    except Exception as e:
        logging.exception("[LivedoorPost] 例外発生")
        return {"ok": False, "error": str(e)}

def extract_post_url(xml_response):
    match = re.search(r"<link rel=['\"]alternate['\"] type=['\"]text/html['\"] href=['\"](.*?)['\"]", xml_response)
    return match.group(1) if match else None
