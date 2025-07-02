# app/services/blog_signup/note_signup.py
import logging, secrets, string, asyncio, re
from datetime import datetime
from flask import current_app
from app import db
from app.models import ExternalBlogAccount, BlogType
from .crypto_utils import encrypt
from app.services.email_utils import create_temp_mail, wait_for_link
from playwright.async_api import async_playwright

NOTE_SIGNUP_URL = "https://note.com/signup"

def _rand_pw(n: int = 12):
    return "".join(secrets.choice(string.ascii_letters+string.digits) for _ in range(n))

async def _signup(email: str, pw: str, username: str) -> None:
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        pg = await br.new_page()
        await pg.goto(NOTE_SIGNUP_URL)
        await pg.fill('input[type="email"]', email)
        await pg.fill('input[name="password"]', pw)
        await pg.fill('input[name="name"]', username)
        await pg.click('button[type="submit"]')
        await pg.wait_for_timeout(2000)  # Note 側の送信完了待ち
        await br.close()

def signup_note_account(site_id: int):
    """
    1. 使い捨てメール作成
    2. Note サインアップ実行
    3. 認証メールの URL を踏む
    4. ExternalBlogAccount 保存 (status=active)
    """
    email, mail_pw, token = create_temp_mail()
    note_pw   = _rand_pw()
    username  = "user" + secrets.token_hex(4)

    # ② Note signup フォーム送信
    asyncio.run(_signup(email, note_pw, username))

    # ③ 認証リンクを取得してクリック
    link = wait_for_link(token, pattern=r"https://note\.com/email/.+")
    if not link:
        raise RuntimeError("認証メールが取得できませんでした")
    # Playwright でリンクを開けば認証完了
    asyncio.run(_open_link(link))

    # ④ DB 保存

    acct = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.NOTE,
        email     = encrypt(email),
        password  = encrypt(note_pw),
        username  = username,
        status    = "active",
        created_at= datetime.utcnow()
    )
    db.session.add(acct)
    db.session.commit()
    current_app.logger.info(f"[NoteSignup] New account {email}")
    return acct

# --- 認証リンクを開くだけのヘッドレスブラウザ ---
async def _open_link(url: str):
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        pg = await br.new_page()
        await pg.goto(url)
        await pg.wait_for_timeout(1500)
        await br.close()
