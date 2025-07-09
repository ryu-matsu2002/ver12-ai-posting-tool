# scripts/label_captcha.py

import csv
import os
from pathlib import Path
from PIL import Image
import pytesseract

# ── 設定 ───────────────────────────────────────────────
DATASET_DIR  = Path("dataset/raw")
LABEL_CSV    = Path("dataset/labels.csv")
NUM_TARGETS  = 200          # ラベル付けする枚数
TESS_CONFIG  = "--psm 7 -l jpn"   # 1 行テキスト想定
# ────────────────────────────────────────────────────

def guess_text(image_path: Path) -> str:
    """Tesseract でひらがな推測（5 文字まで切り出す）"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config=TESS_CONFIG)
        guess = "".join([c for c in text if '\u3041' <= c <= '\u309F']).strip()
        return guess[:5]
    except Exception:
        return ""

def main():
    image_files = sorted(DATASET_DIR.glob("*.png"))[:NUM_TARGETS]
    if not image_files:
        print("❌ dataset/raw に画像が見つかりません")
        return

    print(f"🖼️ {len(image_files)} 枚をラベリングします（Enter で既定値を採用）")
    with LABEL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for img_path in image_files:
            try:
                Image.open(img_path).show()
            except Exception:
                pass  # GUI の無い環境ならスキップ

            default = guess_text(img_path)
            prompt  = f"[{default}] >>> {img_path.name} の文字: "
            label   = input(prompt).strip() or default

            if label == "":
                print("⚠️ 空ラベル → スキップ")
                continue

            writer.writerow([img_path.name, label])
            print("✅  保存しました")

    print(f"\n🎉 ラベル付け完了 → {LABEL_CSV}")

if __name__ == "__main__":
    main()
