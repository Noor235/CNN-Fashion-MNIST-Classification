import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
import sys, os
sys.path.append(os.path.dirname(__file__))
from task3_model_design import Model3

# ── Custom Dataset with Augmentation ─────────────────────────────────
class AugmentedDataset(Dataset):
    def __init__(self, x, y, augment=True):
        self.x = x
        self.y = y
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = torch.tensor(self.x[idx], dtype=torch.float32)
        if self.augment:
            img = self.transform(img)
        return img, self.y[idx]


# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")
d = np.load('../data.npz')
x_train = d['x_train']
y_train = torch.tensor(d['y_train'], dtype=torch.long)
x_test  = torch.tensor(d['x_test'],  dtype=torch.float32)
y_test  = torch.tensor(d['y_test'],  dtype=torch.long)

val_size = int(0.1 * len(x_train))
x_val, y_val = x_train[:val_size], y_train[:val_size]
x_tr,  y_tr  = x_train[val_size:], y_train[val_size:]

train_dataset = AugmentedDataset(x_tr, y_tr, augment=True)
val_dataset   = AugmentedDataset(x_val, y_val, augment=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Train improved Model 3 ───────────────────────────────────────────
model     = Model3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

EPOCHS = 8
print(f"\nTraining Improved Model 3 (with Data Augmentation + LR Scheduler)")
print("="*60)

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(xb)
        train_correct += (out.argmax(1) == yb).sum().item()
        train_total   += len(xb)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            val_loss    += loss.item() * len(xb)
            val_correct += (out.argmax(1) == yb).sum().item()
            val_total   += len(xb)

    scheduler.step()
    t_acc  = train_correct / train_total
    v_acc  = val_correct   / val_total
    t_loss = train_loss    / train_total
    v_loss = val_loss      / val_total
    print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {t_loss:.4f} — acc: {t_acc:.4f} "
          f"— val_loss: {v_loss:.4f} — val_acc: {v_acc:.4f}")

# ── Test accuracy ─────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    out    = model(x_test.to(device))
    y_pred = out.argmax(1).cpu()
    acc    = (y_pred == y_test).float().mean().item()

print(f"\nImproved Model 3 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
torch.save(model.state_dict(), '../m3_improved.pth')
print("Saved m3_improved.pth")

# ── Improvements Summary ──────────────────────────────────────────────
print("\n" + "="*50)
print("  Improvements Applied:")
print("  ✅ Data Augmentation (flip, rotate, shift)")
print("  ✅ Dropout (0.5 and 0.3)")
print("  ✅ Batch Normalization")
print("  ✅ Learning Rate Scheduler (StepLR)")
print("="*50)
