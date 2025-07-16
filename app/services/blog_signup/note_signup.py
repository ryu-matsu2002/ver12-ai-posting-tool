# app/services/blog_signup/note_signup.py

"""
Noteブログ アカウント自動登録（GPTエージェント仕様）
"""

import logging
import time
import random
import string
import asyncio
from app.models import ExternalBlogAccount
from app.enums import BlogType
from app import db
from app.services.mail_utils.mail_gw import create_inbox
from app.services.blog_signup.crypto_utils import encrypt
from app.services.agents.note_gpt_agent import run_note_signup

logger = logging.getLogger(__name__)

def generate_safe_id(n=10):
    chars = string.ascii_lowercase + string.digits + "_"
    return ''.join(random.choices(chars, k=n))

def register_blog_account(site, email_seed="note") -> ExternalBlogAccount:
    import nest_asyncio
    nest_asyncio.apply()

    # すでにアカウントが存在する場合はスキップ
    account = ExternalBlogAccount.query.filter_by(
        site_id=site.id, blog_type=BlogType.NOTE
    ).first()
    if account:
        return account

    email, token = create_inbox()
    logger.info("[NOTE-Signup] disposable email = %s", email)

    password = "Nt" + str(int(time.time()))
    nickname = generate_safe_id(10)

    try:
        asyncio.run(run_note_signup(email, nickname, password))
    except Exception as e:
        logger.error("[NOTE-Signup] failed: %s", str(e))
        raise

    new_account = ExternalBlogAccount(
        site_id=site.id,
        blog_type=BlogType.NOTE,
        email=email,
        username=nickname,
        password=password,
        atompub_key_enc=None,
        api_post_enabled=False,
        nickname=nickname
    )
    db.session.add(new_account)
    db.session.commit()
    return new_account

def signup(site, email_seed="note"):
    return register_blog_account(site, email_seed=email_seed)
