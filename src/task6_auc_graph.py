import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import sys, os
sys.path.append(os.path.dirname(__file__))
from task3_model_design import Model3

# ── Load data ────────────────────────────────────────────────────────
d = np.load('../data.npz')
x_test = torch.tensor(d['x_test'], dtype=torch.float32)
y_test = d['y_test']

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load Model 3 ─────────────────────────────────────────────────────
model = Model3()
model.load_state_dict(torch.load('../m3.pth', map_location=device))
model.eval()

with torch.no_grad():
    y_prob = model(x_test.to(device)).cpu().numpy()

# Binarize labels
y_test_bin = label_binarize(y_test, classes=list(range(10)))

# ── ROC / AUC Curve per class ─────────────────────────────────────────
colors = plt.cm.tab10.colors
plt.figure(figsize=(12, 8))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i],
             label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0,1],[0,1],'k--', linewidth=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curve — AUC per Class (Model 3)', fontsize=15, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../auc_roc_curve.png', dpi=150)
plt.show()
print("Saved auc_roc_curve.png")

macro_auc = roc_auc_score(y_test_bin, y_prob, average='macro')
print(f"Macro Average AUC: {macro_auc:.4f}")
