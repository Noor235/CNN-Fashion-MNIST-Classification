import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
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

# ── Comparison Table ─────────────────────────────────────────────────
print("\n" + "="*85)
print(f"{'Model':<10}{'TrainAcc':<10}{'ValAcc':<10}{'TrainLoss':<12}"
      f"{'ValLoss':<10}{'Prec':<8}{'Recall':<8}{'F1':<8}")
print("="*85)

all_preds = {}
for name, ModelClass, pth in models_cfg:
    model = ModelClass
    model.load_state_dict(torch.load(pth, map_location=device))
    model.eval()

    with torch.no_grad():
        out    = model(x_test.to(device))
        y_pred = out.argmax(1).cpu().numpy()

    h    = np.load(pth.replace('.pth','_hist.npz'))
    p    = precision_score(y_test, y_pred, average='weighted')
    r    = recall_score(y_test, y_pred, average='weighted')
    f    = f1_score(y_test, y_pred, average='weighted')
    tacc = h['train_acc'][-1]
    vacc = h['val_acc'][-1]
    tloss= h['train_loss'][-1]
    vloss= h['val_loss'][-1]

    print(f"{name:<10}{tacc:<10.4f}{vacc:<10.4f}{tloss:<12.4f}"
          f"{vloss:<10.4f}{p:<8.3f}{r:<8.3f}{f:<8.3f}")
    all_preds[name] = y_pred

print("="*85)

# ── Accuracy & Loss Curves (Model 3) ─────────────────────────────────
h = np.load('../m3_hist.npz')
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(h['train_acc'], label='Train Accuracy', marker='o')
plt.plot(h['val_acc'],   label='Val Accuracy',   marker='s')
plt.title('Model 3 — Accuracy Curve', fontsize=13)
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(h['train_loss'], label='Train Loss', marker='o')
plt.plot(h['val_loss'],   label='Val Loss',   marker='s')
plt.title('Model 3 — Loss Curve', fontsize=13)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('../accuracy_loss_curves.png', dpi=150)
plt.show()
print("Saved accuracy_loss_curves.png")

# ── Confusion Matrix (Model 3) ────────────────────────────────────────
y_pred3 = all_preds["Model 3"]
cm = confusion_matrix(y_test, y_pred3)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — Model 3 (Best)', fontsize=14)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../confusion_matrix.png', dpi=150)
plt.show()
print("Saved confusion_matrix.png")

# ── Classification Report ─────────────────────────────────────────────
print("\n===== Classification Report — Model 3 =====")
print(classification_report(y_test, y_pred3, target_names=class_names))
