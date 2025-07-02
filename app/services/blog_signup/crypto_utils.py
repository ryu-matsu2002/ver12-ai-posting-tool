# app/services/blog_signup/crypto_utils.py
"""
外部ブログアカウントのメール / パスワードを
暗号化・複合化するユーティリティ
"""

import os
import logging
from cryptography.fernet import Fernet, InvalidToken

# ─────────────────────────────────────────────
# 鍵の生成 / 取得
#   環境変数 BLOG_SECRET_KEY があればそれを使用。
#   無い場合は自動生成して同じプロセス内で利用する。
# ─────────────────────────────────────────────
_SECRET = os.getenv("BLOG_SECRET_KEY")
if not _SECRET:
    _SECRET = Fernet.generate_key().decode()
    os.environ["BLOG_SECRET_KEY"] = _SECRET

fernet = Fernet(_SECRET.encode())

# ───────────────────────────
# 公開関数
# ───────────────────────────
def encrypt(text: str) -> str:
    """
    プレーン文字列を暗号化して返す
    """
    return fernet.encrypt(text.encode()).decode()


def decrypt(token: str) -> str:
    """
    暗号化文字列を複合化して返す。
    ・過去に「平文をそのままDBへ保存してしまった」行がある場合、
      decrypt() で InvalidToken が出る。
    ・本関数では例外を握りつぶし、平文をそのまま返すことで
      アプリ全体を止めないフェイルセーフ設計とする。
    """
    try:
        return fernet.decrypt(token.encode()).decode()
    except InvalidToken:
        logging.warning(f"[crypto_utils] InvalidToken → 平文扱いで返却: {token[:8]}...")
        return token
