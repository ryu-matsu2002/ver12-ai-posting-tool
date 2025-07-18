# scripts/train_captcha.py

import csv
from pathlib import Path
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ─── 設定 ───────────────────────────────────
CSV_PATH    = Path("dataset/labels.csv")
IMG_DIR     = Path("dataset/raw")
FAILED_DIR  = Path("captcha_failed")
MODEL_DIR   = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
IMG_W, IMG_H = 200, 60
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─── ひらがな46音 ────────────────────────────
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
)
IDX2CHAR = ["_"] + HIRAGANA  # 0=blank
CHAR2IDX = {c: i for i, c in enumerate(IDX2CHAR)}

# ─── Dataset ────────────────────────────────
class CaptchaSet(Dataset):
    def __init__(self, csv_path: Path, failed_dir: Path = None):
        self.records = []

        # 通常データセット
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    lab = row["label"].strip()
                    if len(lab) == 5 and all(ch in CHAR2IDX for ch in lab):
                        self.records.append((IMG_DIR / row["filename"], lab))

        # 失敗画像からの追加学習
        if failed_dir and failed_dir.exists():
            for path in failed_dir.glob("ld_*.png"):
                m = re.search(r"ld_([ぁ-ん]{5})", path.stem)
                if m:
                    label = m.group(1)
                    if all(c in CHAR2IDX for c in label):
                        self.records.append((path, label))

        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
        ])

    def __len__(self):  return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        y = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return img, y

# ─── モデル定義 ──────────────────────────────
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes=len(IDX2CHAR)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3),
            nn.AdaptiveAvgPool2d((1, 5)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                      # [B,128,1,5]
        feat = feat.squeeze(2).permute(0, 2, 1) # [B,5,128]
        return self.fc(feat)                    # [B,5,C]

# ─── 学習ループ ──────────────────────────────
def train():
    ds = CaptchaSet(CSV_PATH, FAILED_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleCRNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    cri = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = cri(out.view(-1, len(IDX2CHAR)), y.view(-1))
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep}/{EPOCHS}  loss={tot/len(dl):.4f}")

    dst = MODEL_DIR / "captcha_crnn.pth"
    torch.save(model.state_dict(), dst)
    print("✅ saved:", dst)

if __name__ == "__main__":
    train()
