import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from autoencoder import EnvelopeDataset  # train_test_loop,
from autoencoder import (
    envelope_train_loader,
    envelope_valid_loader,
    save_features,
    train_test_loop,
    waveform_plotter,
)


class NewAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=2, padding=5),
            # nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=9, stride=2, padding=4),
            # nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=7, stride=2, padding=3),
            # nn.ReLU(),
            nn.Conv1d(2, 1, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            # nn.Conv1d(2, 1, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
        )
        self.up_conv = nn.Sequential(
            # nn.ConvTranspose1d(
            #     1,
            #     2,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     output_padding=0,
            # ),
            # nn.ReLU(),
            nn.ConvTranspose1d(
                1, 2, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            # nn.ReLU(),
            nn.ConvTranspose1d(
                2, 4, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            # nn.ReLU(),
            nn.ConvTranspose1d(
                4, 8, kernel_size=9, stride=2, padding=4, output_padding=1
            ),
            # nn.ReLU(),
            nn.ConvTranspose1d(
                8, 1, kernel_size=11, stride=2, padding=5, output_padding=1
            ),
        )
        self.linear = nn.Linear(375, 188)

    def down(self, x):
        return self.linear(self.down_conv(x))

    def up(self, x):
        return self.up_conv(torch.matmul(x, self.linear.weight))

    def forward(self, x):
        return self.up(self.down(x))


def custom_loss_fn(model, X, epoch):
    enc = model.down(X)
    rec = model.up(enc)
    rl = nn.MSELoss()(X, rec)
    if epoch < 10:
        return rl
    L = enc.shape[-1]
    C = torch.corrcoef(enc[:, 0, :].T)
    assert C.shape == (L, L)
    # cl = C.abs().sum() / (L * L)
    cl = torch.square(C).sum() / (L * L)
    print(f"{rl=:0.6f}, {cl=:0.6f}")
    return rl + 0.01 * cl
