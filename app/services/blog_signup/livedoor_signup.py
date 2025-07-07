# -*- coding: utf-8 -*-
"""
完全自動：Livedoor ID 登録 → ブログ開設 → AtomPub APIキー発行
"""

from __future__ import annotations
import logging, random, string
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from app.services.mail_utils.mail_tm import create_inbox, poll_latest_link
from app.models import db, ExternalBlogAccount, BlogType

REG_URL      = "https://member.livedoor.com/lite/register/"
BLOG_CREATE  = "https://member.livedoor.com/blog/register"
APIKEY_URL   = "https://livedoor.blogcms.jp/atompub/{blog}/apikey"


# ---------------------------------------------------------------- helpers
def _rand_pw(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ---------------------------------------------------------------- main
def signup(site_id: int) -> ExternalBlogAccount:
    """site_id にひも付く Livedoor ブログを完全自動生成し ExternalBlogAccount を返す"""
    email, jwt = create_inbox()
    passwd     = _rand_pw()
    nick       = "auto" + datetime.utcnow().strftime("%f")  # ID / blog 共通

    with sync_playwright() as p:
        br   = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = br.new_page()
        page.set_default_timeout(60_000)

        # ① 仮登録（メール送信）
        page.goto(REG_URL, wait_until="networkidle")
        page.wait_for_selector("input[type='email']")
        page.fill("input[type='email']", email)
        page.click("button[type='submit']")

        # ② メール確認リンク → 本登録ページ
        verify = poll_latest_link(jwt, sender_like="@livedoor", timeout=300)
        if not verify:
            br.close()
            raise RuntimeError("verification mail not found")
        page.goto(verify, wait_until="networkidle")

        # ③ ユーザー情報入力（/register/input）
        page.wait_for_selector("#username")
        page.fill("#username", nick)
        page.fill("input[name='password']",  passwd)
        page.fill("input[name='password2']", passwd)
        page.fill("input[name='mail']",      email)
        # ⬇⬇⬇ ボタンセレクタを修正 ⬇⬇⬇
        page.click("input[value='ユーザー情報を登録']")

        # ④ ブログ開設
        page.goto(BLOG_CREATE, wait_until="networkidle")
        page.wait_for_selector("input[name='blog_id']")
        page.fill("input[name='blog_id']", nick)
        page.fill("input[name='title']",   f"{nick}-blog")
        page.check("input[type='checkbox']")
        page.click("input[type='submit']")

        # ⑤ AtomPub APIキー発行
        page.goto(APIKEY_URL.format(blog=nick), wait_until="networkidle")
        try:
            page.click("text=APIキーを発行", timeout=8_000)
        except PWTimeout:
            pass
        api_key = page.text_content("css=td.api_key, css=code").strip()
        br.close()

    # ⑥ DB 保存
    acc = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.LIVEDOOR,
        email     = email,
        username  = nick,
        password  = passwd,
        nickname  = nick,
        status    = "active",
        message   = api_key,
    )
    db.session.add(acc)
    db.session.commit()
    logging.info("[LD-Signup] account id=%s created", acc.id)
    return acc
