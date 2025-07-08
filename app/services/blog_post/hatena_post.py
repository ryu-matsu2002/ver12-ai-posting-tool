# -*- coding: utf-8 -*-
"""
Hatena Blog - 記事投稿ユーティリティ
success → {"ok":True,"url":...,"posted_at":datetime}
error   → {"ok":False,"error":...}
"""
from __future__ import annotations
import asyncio, json, logging, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from .. import db
from app.models import ExternalBlogAccount

_POST_URL = "https://blog.hatena.ne.jp/-/draft"

async def _async_post(acct: ExternalBlogAccount,
                      title:str, body_html:str) -> str:
    if not acct.cookie_path:
        raise RuntimeError("cookie_path missing")

    ck = Path(acct.cookie_path)
    if not ck.exists():
        raise FileNotFoundError(f"cookie file not found: {ck}")

    async with async_playwright() as p:
        br  = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = await br.new_context(storage_state=json.loads(ck.read_text()),
                                   locale="ja-JP")
        pg  = await ctx.new_page()
        await pg.goto(_POST_URL, timeout=30_000)
        # タイトル
        await pg.fill("input[name='entry[title]']", title)
        # 本文は CodeMirror → textarea[name='entry[body]']
        await pg.fill("textarea[name='entry[body]']", body_html)
        # 公開
        await pg.click("button#publishBtn")
        # 公開後 URL を取得
        await pg.wait_for_url(re.compile(r"https://.*\.hatenablog\.com/entry/.*"), timeout=60_000)
        url = pg.url
        await br.close()
        return url

def post_hatena_article(account: ExternalBlogAccount,
                        title:str, body_html:str,
                        image_path:Optional[str]=None) -> Dict[str,object]:
    try:
        url = asyncio.run(_async_post(account, title, body_html))
        account.posted_cnt += 1
        db.session.commit()
        return {"ok": True, "url": url, "posted_at": datetime.now(timezone.utc)}
    except (PWTimeout, Exception) as e:
        logging.exception("[HatenaPost] failed: %s", e)
        return {"ok": False, "error": str(e)}
