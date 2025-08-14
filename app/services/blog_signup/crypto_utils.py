"""
外部ブログアカウントのメール / パスワード / APIキーを
暗号化・復号するユーティリティ
"""

import os
import logging
from cryptography.fernet import Fernet, InvalidToken

# ─────────────────────────────────────────────
# 鍵の取得（必須）
#   ※ .env などで BLOG_SECRET_KEY を必ず固定配布してください
# ─────────────────────────────────────────────
_SECRET = os.getenv("BLOG_SECRET_KEY")
if not _SECRET:
    # ここで自動生成してしまうと再起動毎に鍵が変わり復号不能になる
    raise RuntimeError(
        "BLOG_SECRET_KEY is not set. Please set a stable key in environment (e.g. .env)."
    )

fernet = Fernet(_SECRET.encode())

# ───────────────────────────
# 公開関数
# ───────────────────────────
def encrypt(text: str) -> str:
    """プレーン文字列を暗号化して返す"""
    if text is None:
        raise ValueError("encrypt() requires non-empty text")
    return fernet.encrypt(text.encode()).decode()


def decrypt(token: str, *, strict: bool = True) -> str:
    """
    暗号化文字列を復号して返す。

    strict=True  : 復号失敗時に例外を送出（推奨。誤った平文フォールバックを禁止）
    strict=False : 復号失敗時は警告ログを出して空文字を返す（互換用）
                   ※ 旧実装の「そのまま token を返す」は危険なのでやめる
    """
    try:
        return fernet.decrypt(token.encode()).decode()
    except InvalidToken:
        if strict:
            raise ValueError("Invalid encrypted token (cannot be decrypted with BLOG_SECRET_KEY)")
        logging.warning("[crypto_utils] InvalidToken -> return empty string (lenient mode)")
        return ""


def decrypt_lenient(token: str) -> str:
    """
    互換用の緩い復号。失敗時は空文字を返す（明示的にこれを選んだ場合のみ）
    """
    return decrypt(token, strict=False)
