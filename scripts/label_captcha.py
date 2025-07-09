# scripts/label_captcha.py
"""
CAPTCHA ラベル付け：OCR が自動候補を出すので
Enter で確定、違えば修正して Enter。
画像プレビュー無しなので SSH ターミナルだけで完結します。
"""

import csv
from pathlib import Path
import pytesseract
from PIL import Image
import shutil, pytesseract
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"

# ── 設定 ──────────────────────────────
DATASET_DIR = Path("dataset/raw")
LABEL_CSV   = Path("dataset/labels.csv")
NUM_TARGETS = 200                 # 200 枚で十分
TESS_CFG    = "--psm 7 -l jpn"    # 1 行テキスト想定
# ────────────────────────────────────

def ocr_guess(img_path: Path) -> str:
    """Tesseract で ひらがな予測（先頭5文字）"""
    text = pytesseract.image_to_string(Image.open(img_path), config=TESS_CFG)
    hiragana = [c for c in text if "\u3041" <= c <= "\u309F"]
    return "".join(hiragana)[:5]

def main():
    imgs = sorted(DATASET_DIR.glob("*.png"))[:NUM_TARGETS]
    if not imgs:
        print("❌ dataset/raw に画像がありません")
        return

    print(f"🔖  {len(imgs)} 枚に OCR 推測を付けます。Enter = そのまま採用")
    with LABEL_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])

        for i, img in enumerate(imgs, 1):
            guess = ocr_guess(img)
            label = input(f"[{i}/{len(imgs)}] {img.name}  推測 → '{guess}' : ").strip() or guess
            if label == "":
                print("⚠️ 空なのでスキップ")
                continue
            w.writerow([img.name, label])

    print(f"\n🎉  完了！ → {LABEL_CSV}")

if __name__ == "__main__":
    main()
