# app/services/blog_signup/crypto_utils.py
import os
from cryptography.fernet import Fernet

# 環境変数 `BLOG_SECRET_KEY` があればそれを使い、なければ生成
_SECRET = os.getenv("BLOG_SECRET_KEY")
if not _SECRET:
    _SECRET = Fernet.generate_key().decode()
    os.environ["BLOG_SECRET_KEY"] = _SECRET
fernet = Fernet(_SECRET.encode())

def encrypt(text: str) -> str:
    return fernet.encrypt(text.encode()).decode()

def decrypt(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()
