# app/utils/captcha_dataset_utils.py

from pathlib import Path
import shutil

# 保存先ディレクトリ（data配下に構成）
DATASET_DIR = Path("data/captcha_dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def save_captcha_label_pair(image_filename: str, label: str) -> bool:
    """
    CAPTCHA画像と人間の手入力ラベルを保存（PNG + .txt）。
    例:
        static/captchas/captcha_abcd123.png → data/captcha_dataset/captcha_abcd123.png
        label="うるう年" → data/captcha_dataset/captcha_abcd123.txt
    """
    src_path = Path("static/captchas") / image_filename
    dst_img_path = DATASET_DIR / image_filename
    dst_txt_path = DATASET_DIR / (dst_img_path.stem + ".txt")

    if not src_path.exists():
        print(f"[WARN] CAPTCHA元画像が見つかりません: {src_path}")
        return False

    try:
        # 画像コピー
        shutil.copy(src_path, dst_img_path)
        # ラベル保存（UTF-8で1行テキスト）
        dst_txt_path.write_text(label.strip(), encoding="utf-8")
        print(f"[OK] CAPTCHA画像とラベル保存: {dst_img_path}, {dst_txt_path}")
        return True
    except Exception as e:
        print(f"[ERR] CAPTCHA保存に失敗: {e}")
        return False

def save_captcha_label_pair(image_filename: str, label: str) -> bool:
    """
    CAPTCHA画像と人間の手入力ラベルを保存（PNG + .txt）。
    例:
        static/captchas/captcha_abcd123.png → data/captcha_dataset/captcha_abcd123.png
        label="うるう年" → data/captcha_dataset/captcha_abcd123.txt
    """
    src_path = Path("static/captchas") / image_filename
    dst_img_path = DATASET_DIR / image_filename
    dst_txt_path = DATASET_DIR / (dst_img_path.stem + ".txt")

    if not src_path.exists():
        print(f"[WARN] CAPTCHA元画像が見つかりません: {src_path}")
        return False

    try:
        # 画像コピー
        shutil.copy(src_path, dst_img_path)
        # ラベル保存（UTF-8で1行テキスト）
        dst_txt_path.write_text(label.strip(), encoding="utf-8")
        print(f"[OK] CAPTCHA画像とラベル保存: {dst_img_path}, {dst_txt_path}")
        return True
    except Exception as e:
        print(f"[ERR] CAPTCHA保存に失敗: {e}")
        return False
