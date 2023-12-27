import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, n_in, n_out, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(n_in, n_out, kernel_size, stride, padding=padding),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
        )


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = 4
        self.down = nn.Sequential(
            ConvBNReLU(1, 1 * base, 3, 2),
            ConvBNReLU(1 * base, 2 * base, 3, 2),
            ConvBNReLU(2 * base, 3 * base, 3, 2),
            ConvBNReLU(3 * base, 4 * base, 3, 2),
            ConvBNReLU(4 * base, 5 * base, 3, 2),
            ConvBNReLU(5 * base, 6 * base, 3, 2),
        )
        self.classify = nn.Conv1d(6 * base, num_classes, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Insert channel dimension.
        return self.classify(self.down(x))
