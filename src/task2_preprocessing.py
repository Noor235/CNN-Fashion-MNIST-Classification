import pandas as pd
import numpy as np

print("Loading CSV files...")
train_df = pd.read_csv('../csv/fashion-mnist_train.csv')
test_df  = pd.read_csv('../csv/fashion-mnist_test.csv')

y_train = train_df.iloc[:, 0].values
x_train = train_df.iloc[:, 1:].values

y_test = test_df.iloc[:, 0].values
x_test = test_df.iloc[:, 1:].values

# Normalize pixel values from 0-255 to 0-1
x_train = (x_train.astype('float32') / 255.0).reshape(-1, 1, 28, 28)
x_test  = (x_test.astype('float32') / 255.0).reshape(-1, 1, 28, 28)

# One-hot encode labels manually
def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes), dtype='float32')
    result[np.arange(len(y)), y] = 1
    return result

y_train_cat = one_hot(y_train)
y_test_cat  = one_hot(y_test)

np.savez('../data.npz',
         x_train=x_train, x_test=x_test,
         y_train=y_train, y_test=y_test,
         y_train_cat=y_train_cat, y_test_cat=y_test_cat)

print("=" * 40)
print("     PREPROCESSING COMPLETE")
print("=" * 40)
print(f"  x_train shape : {x_train.shape}")
print(f"  x_test shape  : {x_test.shape}")
print(f"  y_train shape : {y_train_cat.shape}")
print(f"  y_test shape  : {y_test_cat.shape}")
print(f"  Pixel range   : {x_train.min():.1f} to {x_train.max():.1f}")
print("=" * 40)
print("Done! Saved data.npz")
