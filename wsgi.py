# wsgi.py

# Flask アプリケーションを作成するために必要なインポート
from app import create_app

# アプリケーションのインスタンスを一度だけ生成
# create_app() は Flask アプリケーションの設定や初期化を行います
app = create_app()

# WSGI サーバー（gunicornなど）でこのアプリケーションを実行する準備が整いました
