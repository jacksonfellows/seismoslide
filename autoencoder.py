import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import seisbench.data
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
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

        # Normalize to a mean of 0 and a standard deviation of 1.
        x -= torch.mean(x)
        x /= torch.std(x) + 1e-8
        # x *= 10  # WTF

        # Normalize to [0,1].
        # x -= x.min()
        # x /= x.max() + 1e-8
        # x *= 1000

        return x

    def __len__(self):
        return len(self.metadata)


class WaveformDataset:
    def __init__(self, split):
        self.seisbench_dataset = seisbench.data.WaveformDataset(f"./pnw_splits/{split}")

    def __getitem__(self, i):
        W = self.seisbench_dataset.get_waveforms(i)
        assert W.shape == (3, 6000)
        return torch.tensor(W, dtype=torch.float32)

    def __len__(self):
        return len(self.seisbench_dataset)


waveform_train_loader = DataLoader(
    WaveformDataset("train"), batch_size=64, shuffle=True
)
waveform_valid_loader = DataLoader(
    WaveformDataset("valid"), batch_size=64, shuffle=True
)

spectrogram_train_loader = DataLoader(
    SpectrogramDataset("train"), batch_size=64, shuffle=True
)
spectrogram_valid_loader = DataLoader(
    SpectrogramDataset("valid"), batch_size=64, shuffle=True
)


class WaveformAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(2, 1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                1, 2, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                2, 4, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                4, 8, kernel_size=9, stride=2, padding=4, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                8, 3, kernel_size=11, stride=2, padding=5, output_padding=1
            ),
            # Is ReLU making it harder to generate symmetric/negative seismograms?
        )

    def forward(self, x):
        return self.up(self.down(x))


class SpectrogramAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Use strides instead of max-pooling while down-sampling?
        self.down = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=3, stride=2, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2,
                4,
                kernel_size=3,
                stride=(1, 2),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, output_padding=(1, 1)),
            nn.ReLU(),
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


# def compare(ae):
#     chunk = next(iter(valid_loader))
#     ae.eval()
#     chunk_pred = ae(chunk)
#     chunk = chunk.detach().numpy()
#     chunk_pred = chunk_pred.detach().numpy()
#     N = 10
#     fig, axs = plt.subplots(N, 2, sharex=True, sharey=True, layout="tight")
#     for i in range(N):
#         vmin, vmax = chunk[i, 0].min(), chunk[i, 0].max()
#         print(f"{vmin=}, {vmax=}")
#         axs[i, 0].imshow(chunk[i, 0], vmin=vmin, vmax=vmax, origin="lower")
#         axs[i, 1].imshow(chunk_pred[i, 0], vmin=vmin, vmax=vmax, origin="lower")
#     plt.show()


def waveform_plotter(W, ax):
    for i in range(3):
        ax.plot(W[i])


def train_test_loop(ae, train_loader, valid_loader, plotter, path, epochs=100):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            train(train_loader, ae, loss_fn, optimizer)
            test(valid_loader, ae, loss_fn)
            # Plot original vs. auto-encoded
            chunk = next(iter(valid_loader))
            ae.eval()
            chunk_enc = ae(chunk).detach().numpy()
            chunk = chunk.detach().numpy()
            N = 4
            fig, axs = plt.subplots(
                N, 2, sharex=True, sharey=True, layout="tight", figsize=(8.5, 11)
            )
            for i in range(N):
                plotter(chunk[i], axs[i, 0])
                plotter(chunk_enc[i], axs[i, 1])
            plt.savefig(f"training_output/{epoch:04d}.png")
            plt.close()
    except KeyboardInterrupt:
        pass
    torch.save(ae, path)


def compute_features(ae, dataset):
    L = len(dataset)
    features = np.zeros((L, 60))
    ae.eval()
    for i in range(L):
        X = ae.down(dataset[i]).flatten().detach().numpy()
        features[i] = X
    return features


def save_features(ae):
    for split in ["train", "valid", "test"]:
        features = compute_features(ae, SpectrogramDataset(split))
        np.save(f"{split}_features", features)


def plot_t_sne(X, y):
    X_emb = TSNE().fit_transform(X)
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y)
    plt.show()
