# scripts/test_captcha.py

"""
CAPTCHA推論テストスクリプト
===========================
ローカルの CAPTCHA 画像に対してモデルを使って推論し、結果を表示します。
"""

import sys
from pathlib import Path
from PIL import Image
from app.services.captcha_solver import solve

def main():
    if len(sys.argv) != 2:
        print("使い方: python scripts/test_captcha.py [CAPTCHA画像ファイル]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"指定されたファイルが存在しません: {image_path}")
        sys.exit(1)

    try:
        img = Image.open(image_path).convert("L")
        with image_path.open("rb") as f:
            img_bytes = f.read()

        result = solve(img_bytes)
        print(f"[✅ 推論結果] {image_path.name} ➜ {result}")
    except Exception as e:
        print(f"[❌ エラー] 推論中に例外が発生しました: {e}")

if __name__ == "__main__":
    main()
