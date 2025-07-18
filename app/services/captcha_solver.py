import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import logging
import io
import datetime
from typing import Union

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/captcha_crnn.pth")
CHARS = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"

# 入力画像のサイズ
IMG_WIDTH = 160
IMG_HEIGHT = 60
MAX_LEN = 5  # CAPTCHAの最大文字数

# ── CRNN モデル定義 ─────────────────────
class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 入力: 1ch → 32ch
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 80x30
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x15
        )
        self.rnn = nn.LSTM(64 * 15, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.view(b, w, c * h)    # [B, W, C*H]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x  # [B, W, num_classes + 1]

# ── 推論準備 ──────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(len(CHARS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── 失敗画像保存関数 ─────────────────────
def save_failed_captcha_image(image_bytes: bytes, suffix: str = "fail"):
    """CAPTCHA失敗画像を保存（学習/分析用）"""
    try:
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("captcha_failed") / f"ld_{suffix}_{dt}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"[CAPTCHA Solver] 失敗画像を保存: {path}")
    except Exception as e:
        logger.warning(f"[CAPTCHA Solver] 画像保存失敗: {e}")

# ── CTCデコード ────────────────────────────
def decode(outputs: torch.Tensor) -> str:
    """CTCデコード（重複除去＆blank除去）"""
    pred = outputs.argmax(dim=2).squeeze(0).tolist()
    decoded = []
    last = -1
    for p in pred:
        if p != last and p != len(CHARS):
            decoded.append(CHARS[p])
        last = p
    return ''.join(decoded)

# ── CAPTCHA解読（image path または bytes） ───────
def solve(image: Union[str, bytes], save_on_fail: bool = True) -> str:
    """
    CAPTCHA画像を解読する。
    :param image: ファイルパス（str）または画像バイト列（bytes）
    :param save_on_fail: True の場合、失敗時に画像を保存する
    :return: 解読された文字列（失敗時は""）
    """
    try:
        image_bytes = None

        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image_bytes = image
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise ValueError("Invalid image input")

        image_tensor = transform(pil_image).unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            outputs = model(image_tensor)
        result = decode(outputs.cpu())

        logger.info(f"[CAPTCHA Solver] result = {result}")

        # ✔ 失敗条件を明確化（文字数が5文字未満など）
        if not result or len(result) != MAX_LEN:
            if image_bytes and save_on_fail:
                save_failed_captcha_image(image_bytes, suffix="lenerr")
            return ""

        return result

    except Exception as e:
        logger.warning(f"[CAPTCHA Solver] failed: {e}")
        # ✔ 失敗時に画像を保存
        if isinstance(image, bytes) and save_on_fail:
            save_failed_captcha_image(image, suffix="exception")
        return ""
