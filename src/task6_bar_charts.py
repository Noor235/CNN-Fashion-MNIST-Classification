import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import sys, os
sys.path.append(os.path.dirname(__file__))
from task3_model_design import Model1, Model2, Model3

# ── Load data ────────────────────────────────────────────────────────
d = np.load('../data.npz')
x_test = torch.tensor(d['x_test'], dtype=torch.float32)
y_test = d['y_test']

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_cfg = [
    ("Model 1", Model1(), "../m1.pth"),
    ("Model 2", Model2(), "../m2.pth"),
    ("Model 3", Model3(), "../m3.pth"),
]

train_accs, val_accs = [], []
train_losses, val_losses = [], []
precisions, recalls, f1s = [], [], []
model_names = []

for name, ModelClass, pth in models_cfg:
    model = ModelClass
    model.load_state_dict(torch.load(pth, map_location=device))
    model.eval()
    with torch.no_grad():
        out    = model(x_test.to(device))
        y_pred = out.argmax(1).cpu().numpy()

    h = np.load(pth.replace('.pth','_hist.npz'))
    model_names.append(name)
    train_accs.append(h['train_acc'][-1])
    val_accs.append(h['val_acc'][-1])
    train_losses.append(h['train_loss'][-1])
    val_losses.append(h['val_loss'][-1])
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1s.append(f1_score(y_test, y_pred, average='weighted'))

x = np.arange(len(model_names))

# ── 1. Accuracy Bar Chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.3
b1 = ax.bar(x - w, train_accs, w, label='Train Accuracy', color='#4C72B0')
b2 = ax.bar(x,     val_accs,   w, label='Val Accuracy',   color='#DD8452')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Accuracy Comparison Across Models', fontsize=15, fontweight='bold')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
ax.bar_label(b1, fmt='%.3f', fontsize=9, padding=3)
ax.bar_label(b2, fmt='%.3f', fontsize=9, padding=3)
plt.tight_layout()
plt.savefig('../bar_accuracy.png', dpi=150)
plt.show()
print("Saved bar_accuracy.png")

# ── 2. Loss Bar Chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.3
b1 = ax.bar(x - w/2, train_losses, w, label='Train Loss', color='#C44E52')
b2 = ax.bar(x + w/2, val_losses,   w, label='Val Loss',   color='#8172B2')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Loss Comparison Across Models', fontsize=15, fontweight='bold')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
ax.bar_label(b1, fmt='%.3f', fontsize=9, padding=3)
ax.bar_label(b2, fmt='%.3f', fontsize=9, padding=3)
plt.tight_layout()
plt.savefig('../bar_loss.png', dpi=150)
plt.show()
print("Saved bar_loss.png")

# ── 3. Precision / Recall / F1 Bar Chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.25
b1 = ax.bar(x - w, precisions, w, label='Precision', color='#4C72B0')
b2 = ax.bar(x,     recalls,    w, label='Recall',    color='#DD8452')
b3 = ax.bar(x + w, f1s,        w, label='F1-Score',  color='#55A868')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score', fontsize=13)
ax.set_title('Precision / Recall / F1 Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
for b in [b1, b2, b3]:
    ax.bar_label(b, fmt='%.3f', fontsize=9, padding=3)
plt.tight_layout()
plt.savefig('../bar_prf1.png', dpi=150)
plt.show()
print("Saved bar_prf1.png")

# ── 4. Per-class F1 for Model 3 ──────────────────────────────────────
model = Model3()
model.load_state_dict(torch.load('../m3.pth', map_location=device))
model.eval()
with torch.no_grad():
    y_pred3 = model(x_test.to(device)).argmax(1).cpu().numpy()
f1_per = f1_score(y_test, y_pred3, average=None)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(class_names, f1_per, color=plt.cm.tab10.colors)
ax.set_ylim(0, 1.1); ax.set_ylabel('F1-Score', fontsize=13)
ax.set_title('Per-Class F1-Score — Model 3 (Best)', fontsize=15, fontweight='bold')
ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../bar_perclass_f1.png', dpi=150)
plt.show()
print("Saved bar_perclass_f1.png")
