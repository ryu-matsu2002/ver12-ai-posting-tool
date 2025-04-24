# wsgi.py
from app import create_app

# アプリを一度だけ生成しておく
app = create_app()
