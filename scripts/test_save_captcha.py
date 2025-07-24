# scripts/test_save_captcha.py
from app.services.captcha_solver import save_failed_captcha_image

# テスト用画像（任意の画像ファイルを読み込む）
with open("tests/sample_captcha.png", "rb") as f:
    image_bytes = f.read()

# 保存してみる（suffixを test に）
save_failed_captcha_image(image_bytes, suffix="test")
