# app/services/blog_signup/note_signup.py
import json, random, string, time, asyncio, re, logging
import requests
from playwright.async_api import async_playwright
from datetime import datetime
from .crypto_utils import encrypt
from app import db
from app.models import ExternalBlogAccount, BlogType

MAIL_TM = "https://api.mail.tm"

def _random_string(n=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _create_temp_mail() -> tuple[str, str]:
    # 1. ドメイン取得
    d = requests.get(MAIL_TM + "/domains").json()["hydra:member"][0]["domain"]
    email = f"{_random_string(12)}@{d}"
    password = _random_string(10)

    # 2. メールボックス作成
    r = requests.post(MAIL_TM + "/accounts", json={"address": email, "password": password})
    r.raise_for_status()

    # 3. トークン取得
    tok = requests.post(MAIL_TM + "/token", json={"address": email, "password": password}).json()["token"]
    return email, password, tok

async def _register_note(email: str, passwd: str) -> tuple[str, str]:
    """Playwright で Note にサインアップして username, display_name を返す"""
    user = f"note_{_random_string(6)}"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://note.com/signup")
        # メール入力
        await page.fill('input[type="email"]', email)
        await page.click('button[type="submit"]')
        await page.fill('input[name="password"]', passwd)
        await page.fill('input[name="name"]', user)
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(2000)
        await browser.close()
    return user, passwd

def _wait_for_verify_link(token: str, timeout=90) -> str:
    """mail.tm で認証メールをポーリングしてリンクを返す"""
    headers = {"Authorization": f"Bearer {token}"}
    end = time.time() + timeout
    while time.time() < end:
        msgs = requests.get(MAIL_TM + "/messages", headers=headers).json()["hydra:member"]
        if msgs:
            body = requests.get(f"{MAIL_TM}/messages/{msgs[0]['id']}", headers=headers).json()["html"][0]
            m = re.search(r'https://note\.com/.*?verify.*?"', body)
            if m:
                return m.group(0).rstrip('"')
        time.sleep(3)
    raise TimeoutError("認証メールが届きませんでした")

def _click_verify_link(link: str):
    """認証リンクを GET """
    requests.get(link, timeout=10)

def register_note_account(site_id: int):
    """外部SEO用 Note アカウントを作成し DB に保存"""
    try:
        email, mail_pw, mail_tok = _create_temp_mail()
        note_user, note_pw = asyncio.run(_register_note(email, _random_string(10)))
        verify_link = _wait_for_verify_link(mail_tok)
        _click_verify_link(verify_link)

        acct = ExternalBlogAccount(
            site_id    = site_id,
            blog_type  = BlogType.NOTE,
            email      = encrypt(email),
            username   = note_user,
            password   = encrypt(note_pw),
            status     = "active",
            created_at = datetime.utcnow()
        )
        db.session.add(acct)
        db.session.commit()
        logging.info(f"[NoteSignup] 成功: {note_user}")
        return acct
    except Exception as e:
        logging.exception(f"[NoteSignup] 失敗: {e}")
        acct = ExternalBlogAccount(
            site_id   = site_id,
            blog_type = BlogType.NOTE,
            email     = "",
            username  = "",
            password  = "",
            status    = "error",
            created_at = datetime.utcnow()
        )
        db.session.add(acct)
        db.session.commit()
        return acct
