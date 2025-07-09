# scripts/train_captcha.py
"""
ひらがな CAPTCHA (幅200×高60) を学習する簡易 CRNN スクリプト
- 5 文字固定長を想定
- 文字セット: ひらがな 46 字 + 'blank' = 47 クラス
出力: models/captcha_crnn.pth
"""

import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ───────────── 設定 ─────────────
CSV_PATH = Path("dataset/labels.csv")
IMG_DIR  = Path("dataset/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_W, IMG_H = 200, 60
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────

# ひらがなマッピング
HIRAGANA = [chr(c) for c in range(0x3041, 0x3097)]  # ぁ〜ゖ (46字)
IDX2CHAR = ["_"] + HIRAGANA            # 0 が blank
CHAR2IDX = {c: i for i, c in enumerate(IDX2CHAR)}

# ───────────── Dataset ─────────────
class CaptchaSet(Dataset):
    def __init__(self, csv_path):
        self.records = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                label = row["label"].strip()
                if len(label) != 5:
                    continue          # スキップ
                self.records.append((row["filename"], label))

        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),          # [0,1]
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, label = self.records[idx]
        img = Image.open(IMG_DIR / fname).convert("RGB")
        img = self.tf(img)
        # 文字を index 配列にする
        y = torch.tensor([CHAR2IDX.get(ch, 0) for ch in label], dtype=torch.long)
        return img, y


# ───────────── Model ─────────────
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                    # 30×100
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                    # 15×50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3),                    #  5×16
            nn.AdaptiveAvgPool2d((1, 5))        #  1×5  ←★ NEW
        )
        self.fc = nn.Linear(128, num_classes)   # 高さ=1 なので 128 だけ

    def forward(self, x):
        feat = self.cnn(x)          # [B, 128, 1, 5]
        feat = feat.squeeze(2)      # [B, 128, 5]
        feat = feat.permute(0, 2, 1)  # [B, 5, 128]
        logit = self.fc(feat)         # [B, 5, cls]
        return logit



def train():
    ds = CaptchaSet(CSV_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCRNN(num_classes=len(IDX2CHAR)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for img, y in dl:
            img, y = img.to(DEVICE), y.to(DEVICE)  # y: [B,5]
            optim.zero_grad()
            out = model(img)                       # [B,5,cls]
            loss = criterion(out.view(-1, len(IDX2CHAR)), y.view(-1))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "captcha_crnn.pth")
    print(f"✅ saved → {MODEL_DIR / 'captcha_crnn.pth'}")


if __name__ == "__main__":
    train()
