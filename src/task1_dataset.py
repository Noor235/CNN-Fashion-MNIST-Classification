import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('../csv/fashion-mnist_train.csv')
test_df  = pd.read_csv('../csv/fashion-mnist_test.csv')

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print("=" * 45)
print("        FASHION-MNIST DATASET SUMMARY")
print("=" * 45)
print(f"  Train samples : {len(train_df)}")
print(f"  Test samples  : {len(test_df)}")
print(f"  Total images  : {len(train_df) + len(test_df)}")
print(f"  Classes       : {len(class_names)}")
print(f"  Image size    : 28x28 grayscale")
print(f"  Pixel columns : {train_df.shape[1]-1}")
print("=" * 45)
print("  Classes:")
for i, name in enumerate(class_names):
    print(f"    {i} → {name}")
print("=" * 45)

x_sample = train_df.iloc[:8, 1:].values.reshape(-1, 28, 28)
y_sample  = train_df.iloc[:8, 0].values

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x_sample[i], cmap='gray')
    plt.title(class_names[y_sample[i]], fontsize=10)
    plt.axis('off')
plt.suptitle('Fashion-MNIST Sample Images', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../sample_images.png', dpi=150)
plt.show()
print("Done! Saved sample_images.png")
