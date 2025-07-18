# app/services/captcha_solver/save_failed.py

import os
from datetime import datetime
from shutil import copyfile

SAVE_DIR = "captcha_failed"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_failed_captcha_image(image_path: str, reason: str = "unknown") -> str:
    """
    CAPTCHA失敗画像を保存する（コピーして分類）。
    :param image_path: 保存元の画像ファイルパス
    :param reason: 失敗理由タグ（例："captcha_fail", "mail_fail"）
    :return: 保存先ファイル名（パス付き）
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index = 1
    while True:
        filename = f"{SAVE_DIR}/ld_{timestamp}_{reason}_{index}.png"
        if not os.path.exists(filename):
            break
        index += 1

    try:
        copyfile(image_path, filename)
        return filename
    except Exception as e:
        print(f"[save_failed] 保存に失敗: {e}")
        return ""
