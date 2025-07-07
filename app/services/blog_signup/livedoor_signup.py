# -*- coding: utf-8 -*-
"""
完全自動：Livedoor ID 登録 → ブログ開設 → AtomPub APIキー発行
"""

from __future__ import annotations
import logging, random, string, re
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link
from app.models import db, ExternalBlogAccount, BlogType

REG_URL     = "https://member.livedoor.com/lite/register/"
BLOG_CREATE = "https://member.livedoor.com/blog/register"
APIKEY_URL  = "https://livedoor.blogcms.jp/atompub/{blog}/apikey"

def _rand_pw(n: int = 10) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(n))

def signup(site_id: int) -> ExternalBlogAccount:
    email, jwt = create_inbox()            # ① 使い捨てメール作成
    passwd     = _rand_pw()
    nick       = "auto" + datetime.utcnow().strftime("%f")  # blog と ID 共通

    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = br.new_page()

        # ② 仮登録フォーム（メール送信）
        page.goto(REG_URL)
        page.fill("input[name='mail']", email)
        page.click("input[type=submit]")

        # ③ メールの確認リンク取得 → 本登録
        verify = poll_latest_link(jwt, sender_like="@livedoor.com", timeout=180)
        if not verify:
            raise RuntimeError("verification mail not found")
        page.goto(verify)
        page.fill("input[name='id']", nick)
        page.fill("input[name='pw1']", passwd)
        page.fill("input[name='pw2']", passwd)
        page.click("input[type=submit]")

        # ④ ブログ開設
        page.goto(BLOG_CREATE)
        page.fill("input[name='blog_id']", nick)
        page.fill("input[name='title']", f"{nick}-blog")
        page.check("input[name='agreement']")
        page.click("input[type=submit]")

        # ⑤ AtomPub キー発行
        page.goto(APIKEY_URL.format(blog=nick))
        try:
            page.click("text=APIキーを発行")
        except PWTimeout:
            pass  # 既に発行済み
        api_key = page.text_content("css=td.api_key").strip()
        br.close()

    # ⑥ DB へ保存
    acc = ExternalBlogAccount(
        site_id=site_id,
        blog_type=BlogType.LIVEDOOR,
        email=email,
        username=nick,
        password=passwd,
        nickname=nick,
        status="active",
        message=api_key            # とりあえずAPIキーを message に入れる
    )
    db.session.add(acc); db.session.commit()
    logging.info("[LD-Signup] account id=%s created", acc.id)
    return acc
