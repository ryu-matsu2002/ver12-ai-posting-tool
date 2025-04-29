# ───────────────────────────────────────────────────────────────
# app/wp_poster.py
# ChatGPT で記事を生成し WordPress に投稿するユーティリティ
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import base64
import mimetypes
import logging
from typing import Optional, Tuple
import requests
from openai import OpenAI, BadRequestError

# ─────────────────────────────
# 設定
# ─────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
WP_SITE_URL      = os.getenv("WP_SITE_URL")       # 例：https://example.com
WP_USERNAME      = os.getenv("WP_USERNAME")       # 投稿用ユーザー名
WP_APP_PASSWORD  = os.getenv("WP_APP_PASSWORD")   # アプリケーションパスワード
OPENAI_TIMEOUT   = 120

# Basic‐Auth ヘッダー
def _wp_auth_header() -> dict[str,str]:
    token = base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

# ─────────────────────────────
# ChatGPT で記事生成
# ─────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_article(
    keyword: str,
    title_prompt: str,
    body_prompt: str,
    min_body_chars: int = 1800,
    max_body_chars: int = 3000
) -> Tuple[str,str]:
    """
    タイトルと本文を ChatGPT で生成して返す。
    """
    # タイトル
    sys_title = "あなたはプロの日本語SEOライターです。魅力的なタイトルを1行で返してください。"
    res_title = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":sys_title},
            {"role":"user","content":f"{title_prompt}\n\n▼ キーワード: {keyword}"}
        ],
        max_tokens=80,
        timeout=OPENAI_TIMEOUT
    )
    title = res_title.choices[0].message.content.strip()

    # アウトライン
    sys_outline = "## / ### で見出しを Markdown 形式で返してください。"
    res_outline = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":sys_outline},
            {"role":"user","content":f"{body_prompt}\n\n▼ キーワード: {keyword}\n▼ TITLE: {title}"}
        ],
        max_tokens=400,
        timeout=OPENAI_TIMEOUT
    )
    outline = res_outline.choices[0].message.content

    # 本文
    sys_body = (
        "以下の Markdown アウトラインに沿って、各セクションを550〜750字で日本語の本文を生成してください。"
    )
    res_body = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":sys_body},
            {"role":"user","content":outline}
        ],
        max_tokens=1600,
        timeout=OPENAI_TIMEOUT
    )
    body = res_body.choices[0].message.content.strip()

    # 文字数調整
    if len(body) < min_body_chars:
        body += "\n\nまとめ: この記事の要点を振り返ります。"
    if len(body) > max_body_chars:
        body = body[:max_body_chars].rsplit("。",1)[0] + "。"

    return title, body

# ─────────────────────────────
# WordPress 投稿
# ─────────────────────────────
def post_to_wp(
    title: str,
    content: str,
    status: str = "publish",
    featured_media_id: Optional[int] = None
) -> str:
    """
    WordPress に記事を投稿し、公開 URL を返す。
    """
    api = f"{WP_SITE_URL.rstrip('/')}/wp-json/wp/v2/posts"
    payload = {
        "title": title,
        "content": content,
        "status": status
    }
    if featured_media_id:
        payload["featured_media"] = featured_media_id

    headers = {
        **_wp_auth_header(),
        "Content-Type": "application/json"
    }

    r = requests.post(api, json=payload, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json().get("link","")

# ─────────────────────────────
# 例: 使い方
# ─────────────────────────────
if __name__ == "__main__":
    kw = "海外 ビジネス 旅行"
    tpt = "最新の海外ビジネス渡航に役立つ記事を作成してください。"
    bpt = "読者が飛行機・ホテル手配から現地での業務までスムーズに行えるように案内してください。"

    title, body = generate_article(kw, tpt, bpt)
    try:
        url = post_to_wp(title, body, status="draft")
        print("Successfully posted:", url)
    except Exception as e:
        logging.error("投稿失敗: %s", e)
        raise
