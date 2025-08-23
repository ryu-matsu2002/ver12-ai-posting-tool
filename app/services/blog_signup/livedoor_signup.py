# app/services/blog_signup/livedoor_signup.py
"""
ライブドアブログ アカウント自動登録（ユーティリティ集）
- A案（同一イベントループ固定の常駐コントローラ）採用後は、
  Playwright をここから直接呼ばない。
- ルート → playwright_controller の同期ラッパで実行する。
"""

from __future__ import annotations

import logging
import random
import string
import json
import os
import re as _re
from app import db
from app.enums import BlogType
from app.models import ExternalBlogAccount
from app.services.blog_signup.crypto_utils import encrypt

logger = logging.getLogger(__name__)

# ========== スラッグ生成（blog_id 候補） ==========
try:
    from unidecode import unidecode  # pip install Unidecode（無くても動作）
except Exception:                     # フォールバック
    def unidecode(x): return x

def _slugify_ascii(s: str) -> str:
    """日本語/記号混じり → 半角英数とハイフンの短いスラッグ（livedoor向け、最大20文字）"""
    if not s:
        s = "blog"
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ")
    s = _re.sub(r"[^a-z0-9]+", "-", s)
    s = _re.sub(r"-{2,}", "-", s).strip("-")
    if s and s[0].isdigit():
        s = "blog-" + s
    if not s:
        s = "blog"
    s = s[:20]
    if len(s) < 3:
        s = (s + "-blog")[:20]
    return s

def suggest_livedoor_blog_id(base_text: str, db_session) -> str:
    """
    サイト名/ドメインの文字列から blog_id 候補を作り、
    DBに既存があれば `-2`, `-3`... と採番して一意にする。
    """
    base = _slugify_ascii(base_text)
    candidate, n = base, 0
    while True:
        exists = db_session.query(ExternalBlogAccount.id).filter(
            ExternalBlogAccount.blog_type == BlogType.LIVEDOOR,
            ExternalBlogAccount.livedoor_blog_id == candidate
        ).first()
        if not exists:
            return candidate
        n += 1
        tail = str(n)
        candidate = (base[: max(1, 20 - len(tail) - 1)] + "-" + tail)

# ========== ランダム生成ユーティリティ ==========
def generate_safe_id(n: int = 10) -> str:
    """先頭英字 + 英小文字・数字・アンダーバー"""
    chars = string.ascii_lowercase + string.digits + "_"
    first_char = random.choice(string.ascii_lowercase)
    rest = ''.join(random.choices(chars, k=n - 1))
    return first_char + rest

def generate_safe_password(n: int = 12) -> str:
    """英大小・数字 + 記号（-_%$# のいずれか必須）"""
    chars = string.ascii_letters + string.digits + "-_%$#"
    while True:
        password = ''.join(random.choices(chars, k=n))
        if any(c in "-_%$#" for c in password):
            return password

# ========== 旧・全自動登録API（使用停止） ==========
def register_blog_account(site, email_seed: str = "ld") -> ExternalBlogAccount:
    """
    旧ルートで Playwright を直接呼んでいた全自動登録。
    A案導入後は、/prepare_captcha → /submit_captcha の新フローを使うため、
    ここからは実行しない。
    """
    raise RuntimeError(
        "Deprecated: register_blog_account() は廃止されました。"
        "CAPTCHA フローは /prepare_captcha → /submit_captcha（playwright_controller 経由）を使用してください。"
    )

def signup(site, email_seed: str = "ld"):
    # 互換関数。上と同様に停止。
    return register_blog_account(site, email_seed=email_seed)

# ========== メール本文の認証URL抽出（必要なら使用） ==========
def extract_verification_url(email_body: str) -> str | None:
    """livedoorの認証URLをメール本文から抽出する"""
    import re
    pattern = r"https://member\.livedoor\.com/verify/[a-zA-Z0-9]+"
    m = re.search(pattern, email_body)
    return m.group(0) if m else None

# ========== 一時保存（必要に応じて使用） ==========
TEMP_DIR = "/tmp/livedoor_tasks"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_livedoor_credentials(task_id: str, blog_id: str, api_key: str):
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    with open(path, "w") as f:
        json.dump({"blog_id": blog_id, "api_key": api_key}, f)

def fetch_livedoor_credentials(task_id: str) -> dict | None:
    path = os.path.join(TEMP_DIR, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
