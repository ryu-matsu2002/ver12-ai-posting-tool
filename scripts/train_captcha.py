# scripts/train_captcha.py  ★全文を上書き

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ─── 設定 ───────────────────────────────────
CSV_PATH  = Path("dataset/labels.csv")
IMG_DIR   = Path("dataset/raw")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
IMG_W, IMG_H = 200, 60
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────

# ひらがな 46 音 (ぁ ゃ ょ など除外)
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
assert len(HIRAGANA) == 46
IDX2CHAR = ["_"] + HIRAGANA          # 0=blank
CHAR2IDX = {c: i for i, c in enumerate(IDX2CHAR)}

# ─── Dataset ────────────────────────────────
class CaptchaSet(Dataset):
    def __init__(self, csv_path: Path):
        self.records = []
        with csv_path.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                lab = row["label"].strip()
                if len(lab) == 5 and all(ch in CHAR2IDX for ch in lab):
                    self.records.append((row["filename"], lab))

        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
        ])

    def __len__(self):  return len(self.records)

    def __getitem__(self, idx):
        fname, lab = self.records[idx]
        img = Image.open(IMG_DIR / fname).convert("RGB")
        img = self.tf(img)
        y = torch.tensor([CHAR2IDX[ch] for ch in lab], dtype=torch.long)
        return img, y

# ─── Model ──────────────────────────────────
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes=len(IDX2CHAR)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 30×100
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),   # 15×50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3),  #  5×16
            nn.AdaptiveAvgPool2d((1, 5)),                             #  1×5
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                  # [B,128,1,5]
        feat = feat.squeeze(2).permute(0, 2, 1)  # [B,5,128]
        return self.fc(feat)                # [B,5,num_classes]

# ─── Train ──────────────────────────────────
def train():
    ds = CaptchaSet(CSV_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleCRNN().to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    cri  = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)                        # [B,5,C]
            loss = cri(out.view(-1, len(IDX2CHAR)), y.view(-1))
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {ep}/{EPOCHS}  loss={tot/len(dl):.4f}")

    dst = MODEL_DIR / "captcha_crnn.pth"
    torch.save(model.state_dict(), dst)
    print("✅ saved:", dst)

if __name__ == "__main__":
    train()
