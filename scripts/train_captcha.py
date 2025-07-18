# scripts/train_captcha.py

import csv
from pathlib import Path
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ─── 設定 ─────────────────────────────
CSV_PATH    = Path("dataset/labels.csv")
IMG_DIR     = Path("dataset/raw")
FAILED_DIR  = Path("captcha_failed")
MODEL_PATH  = Path("models/captcha_crnn.pth"); MODEL_PATH.parent.mkdir(exist_ok=True)
IMG_W, IMG_H = 160, 60
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─── ラベル定義 ───────────────────────
CHARS = list(
    "あいうえおかきくけこ"
    "さしすせそたちつてと"
    "なにぬねのはひふへほ"
    "まみむめもやゆよ"
    "らりるれろわをん"
)
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = CHARS
BLANK_IDX = len(CHARS)  # CTC用 blank


# ─── モデル定義（CRNN + CTC対応） ─────
class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 80x30
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 40x15
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2) # 20x7
        )
        self.rnn = nn.LSTM(128 * 7, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc  = nn.Linear(128 * 2, num_classes + 1)  # 256 → +1 for blank

    def forward(self, x):
        x = self.cnn(x)                      # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)            # [B, W, C, H]
        x = x.reshape(b, w, c * h)           # [B, W, C*H]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x  # [B, W, num_classes+1]


# ─── Dataset定義 ──────────────────────
class CaptchaDataset(Dataset):
    def __init__(self):
        self.records = []

        # 通常の学習データ
        if CSV_PATH.exists():
            with CSV_PATH.open(encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    label = row["label"].strip()
                    if len(label) == 5 and all(c in CHAR2IDX for c in label):
                        self.records.append((IMG_DIR / row["filename"], label))

        # 失敗データ（captcha_failed/ld_*.png）
        if FAILED_DIR.exists():
            for path in FAILED_DIR.glob("ld_*.png"):
                m = re.search(r"ld_([ぁ-ん]{5})", path.stem)
                if m:
                    label = m.group(1)
                    if all(c in CHAR2IDX for c in label):
                        self.records.append((path, label))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return img, target, len(target)

# ─── CTC学習ループ ───────────────────
def train():
    ds = CaptchaDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = CRNN(num_classes=len(CHARS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for images, targets, target_lengths in dl:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images)  # [B, W, C]
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [W, B, C]

            input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(dl)
        print(f"Epoch {ep}/{EPOCHS}  Loss = {avg:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ モデル保存:", MODEL_PATH)

# ─── Collate関数（長さ揃え）────────────
def collate(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, labels, lengths

# ─── 実行 ─────────────────────────────
if __name__ == "__main__":
    train()
