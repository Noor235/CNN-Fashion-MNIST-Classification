import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.append(os.path.dirname(__file__))
from task3_model_design import Model1, Model2, Model3

# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")
d = np.load('../data.npz')
x_train = torch.tensor(d['x_train'], dtype=torch.float32)
y_train = torch.tensor(d['y_train'], dtype=torch.long)
x_test  = torch.tensor(d['x_test'],  dtype=torch.float32)
y_test  = torch.tensor(d['y_test'],  dtype=torch.long)

# Validation split (10%)
val_size  = int(0.1 * len(x_train))
x_val, y_val     = x_train[:val_size], y_train[:val_size]
x_tr,  y_tr      = x_train[val_size:], y_train[val_size:]

train_loader = DataLoader(TensorDataset(x_tr, y_tr),   batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val, y_val), batch_size=128)
test_loader  = DataLoader(TensorDataset(x_test, y_test), batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

EPOCHS = 8

def train_model(name, model):
    print(f"\n{'='*50}")
    print(f"  Training {name}")
    print(f"{'='*50}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}

    for epoch in range(EPOCHS):
        # Training
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

        # Validation
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

        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total
        t_loss = train_loss    / train_total
        v_loss = val_loss      / val_total

        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        print(f"  Epoch {epoch+1}/{EPOCHS} — "
              f"loss: {t_loss:.4f} — acc: {t_acc:.4f} — "
              f"val_loss: {v_loss:.4f} — val_acc: {v_acc:.4f}")

    return model, history


# ── Train all 3 models ───────────────────────────────────────────────
models_cfg = [
    ("Model 1", Model1(), "m1"),
    ("Model 2", Model2(), "m2"),
    ("Model 3", Model3(), "m3"),
]

for name, model, fn in models_cfg:
    trained, hist = train_model(name, model)
    torch.save(trained.state_dict(), f"../{fn}.pth")
    np.savez(f"../{fn}_hist.npz", **{k: np.array(v) for k, v in hist.items()})
    print(f"  Saved ../{fn}.pth and ../{fn}_hist.npz")

print("\nAll models trained and saved!")
