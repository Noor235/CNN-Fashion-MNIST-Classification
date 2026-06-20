import torch
import torch.nn as nn

# ── Model 1: Basic CNN with 2 Convolutional Layers ──────────────────
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # Conv Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # Pooling Layer 1
            nn.Conv2d(32, 64, kernel_size=3),  # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # Pooling Layer 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Model 2: CNN with 4 Convolutional Layers ────────────────────────
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # Conv Layer 1
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # Pooling Layer 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv Layer 3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv Layer 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # Pooling Layer 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Model 3: Deep CNN with 5+ Layers + BatchNorm + Dropout ──────────
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # Conv Layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                             # Pooling Layer 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # Conv Layer 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # Conv Layer 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                             # Pooling Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv Layer 5
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    for name, ModelClass in [("Model 1 (2 Conv Layers)", Model1),
                              ("Model 2 (4 Conv Layers)", Model2),
                              ("Model 3 (5+ Conv Layers)", Model3)]:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        model = ModelClass().to(device)
        try:
            summary(model, (1, 28, 28))
        except:
            print(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total:,}")
