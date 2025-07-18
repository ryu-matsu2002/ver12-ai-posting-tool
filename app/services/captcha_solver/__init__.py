# app/services/captcha_solver/__init__.py
"""
ひらがな5文字 CAPTCHA 推論モジュール
使い方:
    from app.services.captcha_solver import solve
    txt = solve(img_bytes)   # img_bytes = bytes (PNG)
"""

from pathlib import Path
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── 文字セット（46 音）─────────────────────────────
HIRAGANA = list(
    "あいうえお"
    "かきくけこ"
    "さしすせそ"
    "たちつてと"
    "なにぬねの"
    "はひふへほ"
    "まみむめも"
    "やゆよ"
    "らりるれろ"
    "わをん"
)  # len=46
IDX2CHAR = ["_"] + HIRAGANA        # 0 が blank
N_CLASS  = len(IDX2CHAR)           # =47

IMG_W, IMG_H = 200, 60

# ── モデル定義（学習時と同一）────────────────────
class SimpleCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * 7, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, 47)  # 47 = len(CHARS)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ── 1 回だけロード ────────────────────────────────
MODEL_PATH = Path("models/captcha_crnn.pth")
DEVICE = "cpu"
_model = SimpleCRNN().to(DEVICE)
_state = torch.load(MODEL_PATH, map_location=DEVICE)
_model.load_state_dict(_state)
_model.eval()

_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

@torch.inference_mode()
def solve(img_bytes: bytes) -> str:
    """PNG バイト列 → ひらがな5文字 推論"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)   # [1,1,60,200]
    logit = _model(x)[0]                   # [5,47]
    idxs  = logit.argmax(dim=-1).tolist()  # [5]
    return "".join(IDX2CHAR[i] for i in idxs)
