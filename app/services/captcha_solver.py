# app/services/captcha_solver.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import logging

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

# ── 推論関数 ──────────────────────────────
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

def decode(outputs):
    """CTCデコード（重複除去＆blank除去）"""
    pred = outputs.argmax(dim=2).squeeze(0).tolist()
    decoded = []
    last = -1
    for p in pred:
        if p != last and p != len(CHARS):
            decoded.append(CHARS[p])
        last = p
    return ''.join(decoded)

def solve(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            outputs = model(image)
        result = decode(outputs.cpu())
        logger.info(f"[CAPTCHA Solver] result = {result}")
        return result
    except Exception as e:
        logger.warning(f"[CAPTCHA Solver] failed: {e}")
        return ""
