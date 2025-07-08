# -*- coding: utf-8 -*-
"""
blog_signup パッケージ公開 API
------------------------------------------------
register_blog_account(site_id, blog_type)
  └ ブログ種別ごとにアカウント登録を振り分ける**同期**関数

変更点（2025-07-06）
────────────────────────────────────────────
* register_note_account_sync() の戻り値を dict → ExternalBlogAccount に変更
* 成功時は acct.status="active" に更新して return
* 失敗時は acct.status="error"／acct.message に理由を残し RuntimeError を raise
  └ 呼び出し側 (_run_external_seo_job) で捕捉→ExternalSEOJob.message へ転記される
"""

from __future__ import annotations

import asyncio
import secrets
import string

from app import db
from app.models import BlogType, ExternalBlogAccount
from .note_signup import signup_note_account  # async coroutine
from .hatena_signup import signup_hatena_account
from .mail_tm_client import create_inbox
# 末尾など分かりやすい場所に
from .livedoor_signup import register_blog_account as register_livedoor_account

# ──────────────────────────────────────────
def _random_email() -> str:
    """example.net ドメインでランダム email を生成"""
    prefix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    return f"{prefix}@example.net"


def _random_password() -> str:
    """8 文字以上・英数＋記号を必ず含むパスワード"""
    core = secrets.token_urlsafe(10)[:8]
    return core + "A1!"


# ──────────────────────────────────────────
def register_note_account_sync(site_id: int) -> ExternalBlogAccount:  # ← 戻り値型をモデルに
    """
    Note アカウントを同期で登録し、成功時は ExternalBlogAccount を返す

    Raises
    ------
    RuntimeError : サインアップ失敗時
    """
    # 1️⃣ DB にレコードを作成
    email    = _random_email()
    password = _random_password()

    acct = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.NOTE,
        email     = email,
        password  = password,
        username  = email.split("@")[0],
        nickname  = email.split("@")[0],
        status    = "pending",
    )
    db.session.add(acct)
    db.session.commit()

    # 2️⃣ Playwright でサインアップ実行
    result = asyncio.run(signup_note_account(acct))

    # 3️⃣ 成否判定
    if result.get("ok"):
        acct.status = "active"
        db.session.commit()
        return acct

    # 失敗時: レコードを error にして例外
    acct.status  = "error"
    acct.message = result.get("error", "unknown")  # ExternalBlogAccount に message カラムがある前提
    db.session.commit()
    raise RuntimeError(acct.message)

# 追記：同期ラッパ
def register_hatena_account_sync(site_id:int) -> ExternalBlogAccount:
    email    = _random_email()
    password = _random_password()
    acct = ExternalBlogAccount(
        site_id   = site_id,
        blog_type = BlogType.HATENA,
        email     = email,
        password  = password,
        username  = email.split("@")[0],
        nickname  = email.split("@")[0],
        status    = "pending",
    )
    db.session.add(acct); db.session.commit()

    result = asyncio.run(signup_hatena_account(acct))
    if result.get("ok"):
        acct.status = "active"; db.session.commit(); return acct
    acct.status = "error"; acct.message = result["error"]; db.session.commit()
    raise RuntimeError(acct.message)


# ──────────────────────────────────────────
def register_blog_account(site_id: int, blog_type: BlogType) -> ExternalBlogAccount:
    """
    ブログ種別ディスパッチャ
    Returns
    -------
    ExternalBlogAccount
        成功時は status="active" のモデル
    """
    if blog_type == BlogType.NOTE:
        return register_note_account_sync(site_id)
    
    elif blog_type == BlogType.HATENA:                            # ★追加
        return register_hatena_account_sync(site_id)

    # まだ未実装のブログ
    raise ValueError(f"Blog type {blog_type} not supported yet.")


__all__ = ["register_blog_account"]
