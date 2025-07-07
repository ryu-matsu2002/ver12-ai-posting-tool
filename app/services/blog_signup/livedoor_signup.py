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

# ---------------------------------------------------------------- URLs
REG_URL      = "https://member.livedoor.com/lite/register/"     # メール入力ページ
BLOG_CREATE  = "https://member.livedoor.com/blog/register"      # ブログ作成ページ
APIKEY_URL   = "https://livedoor.blogcms.jp/atompub/{blog}/apikey"

# ---------------------------------------------------------------- helpers
def _rand_pw(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))

# ---------------------------------------------------------------- main
def signup(site_id: int) -> ExternalBlogAccount:
    """site_id にひも付く Livedoor ブログを完全自動生成し ExternalBlogAccount を返す"""
    email, jwt = create_inbox()                # ① 使い捨てメール
    passwd     = _rand_pw()
    nick       = "auto" + datetime.utcnow().strftime("%f")  # ID / blog 共通

    with sync_playwright() as p:
        br   = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = br.new_page()
        page.set_default_timeout(60_000)       # 全操作 60 秒待機

        # ② 仮登録：メール入力 → 確認メール送信
        page.goto(REG_URL, wait_until="networkidle")
        page.wait_for_selector("input[type='email']")
        page.fill("input[type='email']", email)
        page.click("button[type='submit']")    # ← ボタンは <button>

        # ③ メール内リンク → 本登録フォーム (register/input)
        verify = poll_latest_link(jwt, sender_like="@livedoor", timeout=300)
        if not verify:
            br.close()
            raise RuntimeError("verification mail not found")
        page.goto(verify, wait_until="networkidle")

        # ④ ユーザー情報を入力
        #    実 DOM (2025-07) では：
        #      livedoor ID        → input[name='login'] か #username
        #      パスワード         → input[name='password']
        #      パスワード(確認)    → input[name='password2']
        #      メールアドレス      → input[name='mail']         (自動で入っている)
        page.wait_for_selector("input[name='login'], #username")
        page.fill("input[name='login'], #username", nick)
        page.fill("input[name='password']",  passwd)
        page.fill("input[name='password2']", passwd)
        # メール欄は既に filled されているが念のため
        page.fill("input[name='mail']", email)
        page.click("button[type='submit']")

        # ⑤ ブログ開設
        page.goto(BLOG_CREATE, wait_until="networkidle")
        page.wait_for_selector("input[name='blog_id']")
        page.fill("input[name='blog_id']", nick)
        page.fill("input[name='title']",  f"{nick}-blog")
        page.check("input[type='checkbox']")   # 利用規約
        page.click("input[type='submit']")

        # ⑥ AtomPub API キー発行
        page.goto(APIKEY_URL.format(blog=nick), wait_until="networkidle")
        try:
            page.click("text=APIキーを発行", timeout=8_000)
        except PWTimeout:
            pass                                # 既に発行済み
        api_key = page.text_content("css=td.api_key, css=code").strip()
        br.close()

    # ⑦ DB 保存
    acc = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.LIVEDOOR,
        email     = email,
        username  = nick,
        password  = passwd,
        nickname  = nick,
        status    = "active",
        message   = api_key,        # API キーを message に保管
    )
    db.session.add(acc)
    db.session.commit()
    logging.info("[LD-Signup] account id=%s created", acc.id)
    return acc
