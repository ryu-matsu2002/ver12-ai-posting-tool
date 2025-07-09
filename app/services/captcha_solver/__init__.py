# app/services/captcha_solver/__init__.py
"""
ひらがな 5 文字 CAPTCHA 推論モジュール
使い方:
    from app.services.captcha_solver import solve
    txt = solve(img_bytes)       # img_bytes = bytes (PNG)
"""

from pathlib import Path
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── モデル定義（学習と同じ） ─────────────────────────────
HIRAGANA = [chr(c) for c in range(0x3041, 0x3097)]
IDX2CHAR = ["_"] + HIRAGANA
IMG_W, IMG_H = 200, 60

class SimpleCRNN(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3),
            nn.AdaptiveAvgPool2d((1, 5)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):                 # [B,1,60,200]
        feat = self.cnn(x)                # [B,128,1,5]
        feat = feat.squeeze(2).permute(0, 2, 1)   # [B,5,128]
        return self.fc(feat)              # [B,5,cls]

# ── 1 度だけロード ───────────────────────────────────────
MODEL_PATH = Path("models/captcha_crnn.pth")
_device = "cpu"
_model = SimpleCRNN().to(_device)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
_model.eval()

_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

@torch.inference_mode()
def solve(img_bytes: bytes) -> str:
    """PNG バイト列 → 推論（ひらがな 5 文字）"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(_device)            # [1,1,60,200]
    logit = _model(x)[0]                             # [5,cls]
    pred = logit.argmax(dim=-1).tolist()             # [5]
    return "".join(IDX2CHAR[i] for i in pred)
