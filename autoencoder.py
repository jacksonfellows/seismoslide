import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import seisbench.data
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


def write_spectrograms(split):
    split_dir = f"./pnw_splits/{split}"
    dataset = seisbench.data.WaveformDataset(split_dir)
    spectrogram_path = Path(f"./pnw_splits_spectrograms/{split}")
    for i in range(len(dataset)):
        waveforms = dataset.get_waveforms(i)
        Z = waveforms[0].astype("float32")
        breakpoint()
        _, _, S = scipy.signal.spectrogram(Z)
        S = S.T
        np.save(spectrogram_path / str(i), S)


def write_all():
    for split in ["train", "valid", "test"]:
        write_spectrograms(split)


class SpectrogramDataset:
    def __init__(self, split):
        self.spectrogram_dir = Path(f"./pnw_splits_spectrograms/{split}")
        # Reuse metadata.
        self.metadata = pd.read_csv(f"./pnw_splits/{split}/metadata.csv")

    def __getitem__(self, i):
        x = torch.tensor(np.load(self.spectrogram_dir / f"{i}.npy"))
        x = torch.unsqueeze(x, 0)[:, :, :-1]  # (1, 26, 128)
        return x

    def __len__(self):
        return len(self.metadata)


train_loader = DataLoader(SpectrogramDataset("train"), batch_size=64, shuffle=True)
valid_loader = DataLoader(SpectrogramDataset("valid"), batch_size=64, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            # (batch_size, 1, 26, 128)
            nn.Conv2d(1, 4, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # (batch_size, 4, 13, 64)
            nn.Conv2d(4, 2, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # (batch_size, 2, 13, 32)
            nn.Conv2d(2, 1, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # (batch_size, 1, 6, 16)
        )
        self.up = nn.Sequential(
            # (batch_size, 1, 6, 16)
            nn.Conv2d(1, 1, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Upsample((13, 32)),
            # (batch_size, 1, 13, 32)
            nn.Conv2d(1, 2, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Upsample((13, 64)),
            # (batch_size, 2, 13, 64)
            nn.Conv2d(2, 4, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Upsample((26, 128)),
            # (batch_size, 4, 26, 128)
            nn.Conv2d(4, 1, kernel_size=3, padding="same"),
            # (batch_size, 1, 26, 128)
        )

    def forward(self, x):
        return self.up(self.down(x))


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    n_chunks = len(dataloader)
    for i, X in enumerate(dataloader):
        X_ = model(X)
        loss = loss_fn(X, X_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    n_chunks = len(dataloader)
    for i, X in enumerate(dataloader):
        X_ = model(X)
        loss = loss_fn(X, X_)
        test_loss += loss
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")
    mean_loss = test_loss / n_chunks
    print(f"{mean_loss=:0.5f}")


def compare(ae):
    chunk = next(iter(valid_loader))
    ae.eval()
    chunk_pred = ae(chunk)
    chunk = chunk.detach().numpy()
    chunk_pred = chunk_pred.detach().numpy()
    N = 10
    fig, axs = plt.subplots(N, 2, sharex=True, sharey=True, layout="tight")
    for i in range(N):
        axs[i, 0].imshow(chunk[i, 0], origin="lower")
        axs[i, 1].imshow(chunk_pred[i, 0], origin="lower")
    plt.show()


def go(path, epochs=100):
    ae = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            train(train_loader, ae, loss_fn, optimizer)
            test(valid_loader, ae, loss_fn)
    except KeyboardInterrupt:
        pass
    torch.save(ae, path)
    compare(ae)


def compute_features(ae, dataset):
    L = len(dataset)
    features = np.zeros((L, 96))
    ae.eval()
    for i in range(L):
        X = ae.down(dataset[i]).flatten().detach().numpy()
        features[i] = X
    return features


def save_features(ae):
    for split in ["train", "valid", "test"]:
        features = compute_features(ae, SpectrogramDataset(split))
        np.save(f"{split}_features", features)
