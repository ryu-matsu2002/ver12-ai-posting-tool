import json
from pathlib import Path
from datetime import datetime
import uuid
import os

from app import db
from app.models import ExternalBlogAccount, Site

CONFIG_DIR = Path("scripts/configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_config_json(site_id: int, blog_type: str = "livedoor") -> Path:
    """
    指定サイトIDとブログ種別からGUI用設定JSONを生成する。
    """
    site = Site.query.get(site_id)
    if not site:
        raise ValueError(f"Site ID {site_id} が見つかりません")

    # ランダムファイル名（uuid + timestamp）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = f"{blog_type}_{site_id}_{timestamp}_{uuid.uuid4().hex[:6]}"
    json_path = CONFIG_DIR / f"{config_id}.json"

    # 入力データ（例：livedoor用）
    config = {
        "site_id": site.id,
        "site_name": site.name,
        "url": site.url,
        "blog_type": blog_type,
        "email": f"{uuid.uuid4().hex[:8]}@mail.gw",
        "password": uuid.uuid4().hex[:10],
        "timestamp": timestamp
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return json_path

def start_gui_runner(json_path: Path) -> int:
    """
    xvfb-run + gui_signup_runner.py を起動し、プロセスIDを返す。
    """
    from subprocess import Popen

    cmd = [
        "xvfb-run",
        "--auto-servernum",
        "--server-args='-screen 0 1024x768x24'",
        "python3",
        "scripts/gui_signup_runner.py",
        str(json_path)
    ]
    process = Popen(" ".join(cmd), shell=True)
    return process.pid
