
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()

        # Converting 32x32 to 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # halve the spatial size
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.block1(x)
        return x
