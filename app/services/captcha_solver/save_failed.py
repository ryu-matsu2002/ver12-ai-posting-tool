import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile

FAILED_DIR = Path("captcha_data/failed")
FAILED_DIR.mkdir(parents=True, exist_ok=True)

def save_failed_captcha_image(img_path: str, reason: str = "unknown") -> str:
    """
    失敗したCAPTCHA画像を指定ディレクトリに保存。
    :param img_path: 元画像のパス
    :param reason: 失敗理由タグ（例: bad_prediction / submit_fail など）
    :return: 保存先パス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(img_path).suffix or ".png"
    dest_name = f"{timestamp}_{reason}{suffix}"
    dest_path = FAILED_DIR / dest_name

    if Path(img_path).exists():
        copyfile(img_path, dest_path)
        return str(dest_path)
    else:
        return ""
