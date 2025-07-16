import logging

# ← この設定を先頭に追加
logging.basicConfig(
    level=logging.DEBUG,  # ← WARNINGやDEBUGも出力する
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Flask アプリケーションを作成するために必要なインポート
from app import create_app

# アプリケーションのインスタンスを一度だけ生成
app = create_app()
