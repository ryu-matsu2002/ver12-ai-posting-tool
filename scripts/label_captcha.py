# scripts/label_captcha.py   ← ★全文コピペ

import csv
import shutil
import subprocess
from pathlib import Path
import pytesseract
from PIL import Image

# ── 設定 ─────────────────────────────────────────────
DATASET_DIR  = Path("dataset/raw")
LABEL_CSV    = Path("dataset/labels.csv")
NUM_TARGETS  = 200                       # ラベル付け枚数
TESS_CONFIG  = "--psm 7 -l jpn"
IMG2TXT_BIN  = shutil.which("img2txt")   # caca-utils の実体パス
# ───────────────────────────────────────────────────

def ascii_preview(image_path: Path) -> None:
    """img2txt で画像を ASCII アート表示"""
    if IMG2TXT_BIN:
        subprocess.run([IMG2TXT_BIN, "--gamma=0.6", "--width=60", str(image_path)])
    else:
        print("(img2txt が見つからないため ASCII 表示をスキップ)")

def guess_text(image_path: Path) -> str:
    """Tesseract でひらがな推測（先頭5文字）"""
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
        print("❌ dataset/raw に画像がありません")
        return

    print(f"🖼️ {len(image_files)} 枚をラベリングします（Enter で既定値を採用）")
    with LABEL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for img_path in image_files:
            print(f"\n=== {img_path.name} ===")
            ascii_preview(img_path)

            default = guess_text(img_path)
            label = input(f"[{default}] >>> 文字を入力: ").strip() or default

            if label == "":
                print("⚠️ 空ラベル → スキップ")
                continue

            writer.writerow([img_path.name, label])
            print("✅ 保存しました")

    print(f"\n🎉 完了！ labels.csv を作成しました → {LABEL_CSV}")

if __name__ == "__main__":
    main()
