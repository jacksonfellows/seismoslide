import math
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


class EnvelopeDataset:
    def __init__(self, split):
        self.envelope_dir = Path(f"./pnw_splits_Z_envelopes/{split}")
        # Reuse metadata.
        self.metadata = pd.read_csv(f"./pnw_splits/{split}/metadata.csv")

    def __getitem__(self, i):
        x = torch.tensor(np.load(self.envelope_dir / f"{i}.npy"))
        x = torch.unsqueeze(x, 0)  # (1, 6000)
        x /= x.max() + 1e-8  # Normalize to [0,1].
        return x

    def __len__(self):
        return len(self.metadata)


envelope_train_loader = DataLoader(
    EnvelopeDataset("train"), batch_size=128, shuffle=True
)
envelope_valid_loader = DataLoader(
    EnvelopeDataset("valid"), batch_size=128, shuffle=True
)


class WaveformDataset:
    def __init__(self, split):
        self.seisbench_dataset = seisbench.data.WaveformDataset(f"./pnw_splits/{split}")

    def __getitem__(self, i):
        W = self.seisbench_dataset.get_waveforms(i)
        assert W.shape == (3, 6000)
        W = W[0:1]
        assert W.shape == (1, 6000)
        W /= np.abs(W).max() + 1e-8  # Normalize to [-1,1].
        # TODO: Test the effects of normalization.
        return torch.tensor(W, dtype=torch.float32)

    def __len__(self):
        return len(self.seisbench_dataset)


# waveform_train_loader = DataLoader(
#     WaveformDataset("train"), batch_size=128, shuffle=True
# )
# waveform_valid_loader = DataLoader(
#     WaveformDataset("valid"), batch_size=128, shuffle=True
# )

# spectrogram_train_loader = DataLoader(
#     SpectrogramDataset("train"), batch_size=128, shuffle=True
# )
# spectrogram_valid_loader = DataLoader(
#     SpectrogramDataset("valid"), batch_size=128, shuffle=True
# )


class WaveformAutoencoder(nn.Module):
    def __init__(self, depth, kernel_size, layer_num_channels):
        super().__init__()
        down_layers = []
        up_layers = []
        assert kernel_size % 2 == 1  # Padding only works out for odd-sized kernels.
        length = 6000
        num_channels = 1
        for i in range(depth):
            padding = kernel_size // 2
            num_channels = layer_num_channels[i]
            next_num_channels = layer_num_channels[i + 1]
            down_layers.append(
                nn.Conv1d(
                    num_channels,
                    next_num_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                )
            )
            down_layers.append(nn.ReLU())
            # up_layers will be reversed.
            up_layers.append(
                nn.ConvTranspose1d(
                    next_num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=1 if length % 2 == 0 else 0,
                )
            )
            up_layers.append(nn.ReLU())
            length = math.ceil(length / 2)
            num_channels = next_num_channels
        self.down = nn.Sequential(*down_layers)
        # Drop activation after last layer.
        self.up = nn.Sequential(*list(reversed(up_layers))[1:])

    def forward(self, x):
        return self.up(self.down(x))


class SpectrogramAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.ConvTranspose2d(2, 4, kernel_size=3, stride=(1, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, output_padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up(self.down(x))


def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    n_chunks = len(dataloader)
    for i, X in enumerate(dataloader):
        loss = loss_fn(model, X, epoch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")


def test(dataloader, model, loss_fn, epoch):
    model.eval()
    test_loss = 0
    n_chunks = len(dataloader)
    for i, X in enumerate(dataloader):
        loss = loss_fn(model, X, epoch)
        test_loss += loss
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")
    mean_loss = test_loss / n_chunks
    print(f"{mean_loss=:0.5f}")
    return mean_loss


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
    ax.set_ylim(0, 1)
    for i in range(W.shape[0]):
        ax.plot(W[i])


def train_test_loop(
    ae, train_loader, valid_loader, plotter, path, epochs=100, loss_fn=None
):
    if os.path.exists(path):
        raise ValueError(f"path {path} already exists!")
    if loss_fn is None:
        mse = nn.MSELoss()
        loss_fn = lambda model, X, epoch: mse(X, model.forward(X))
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    chunk = next(iter(valid_loader))
    chunk_det = chunk.detach().numpy()
    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            train(train_loader, ae, loss_fn, optimizer, epoch)
            mean_loss = test(valid_loader, ae, loss_fn, epoch)
            # Plot original vs. auto-encoded
            ae.eval()
            chunk_enc = ae(chunk).detach().numpy()
            N = 6
            fig, axs = plt.subplots(
                N, 2, sharex=True, sharey=True, layout="tight", figsize=(8.5, 11)
            )
            for i in range(N):
                plotter(chunk_det[i], axs[i, 0])
                plotter(chunk_enc[i], axs[i, 1])
            fig.suptitle(f"epoch={epoch}, mean_loss={mean_loss:0.5f}")
            plt.savefig(f"training_output/{epoch:04d}.png")
            plt.close()
    except KeyboardInterrupt:
        pass
    torch.save(ae, path)


def compute_features(ae, dataset):
    L = len(dataset)
    N = len(ae.down(dataset[0]).flatten())
    features = np.zeros((L, N))
    ae.eval()
    for i in range(L):
        X = ae.down(dataset[i]).flatten().detach().numpy()
        features[i] = X
    return features


def save_features(ae, dataset, suffix=""):
    for split in ["train", "valid", "test"]:
        features = compute_features(ae, dataset(split))
        np.save(f"{split}_features{suffix}", features)


def plot_t_sne(X, y):
    X_emb = TSNE().fit_transform(X)
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y)
    plt.show()
