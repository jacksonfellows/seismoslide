import os
from pathlib import Path

import numpy as np
import pandas as pd
import seisbench.data
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from autoencoder import waveform_plotter


# Change EnvelopeDataset to return (X, y) instead of X.
class EnvelopeDataset:
    def __init__(self, split):
        self.envelope_dir = Path(f"./pnw_splits_Z_envelopes/{split}")
        # Reuse metadata.
        self.metadata = pd.read_csv(f"./pnw_splits/{split}/metadata.csv")

    def __getitem__(self, i):
        x = torch.tensor(np.load(self.envelope_dir / f"{i}.npy"))
        x = torch.unsqueeze(x, 0)  # (1, 6000)
        x /= x.max() + 1e-8  # Normalize to [0,1].
        y = int(self.metadata.iloc[i].source_type == "surface event")
        return (x, y)

    def __len__(self):
        return len(self.metadata)


envelope_train_loader = DataLoader(
    EnvelopeDataset("train"), batch_size=128, shuffle=True
)
envelope_valid_loader = DataLoader(
    EnvelopeDataset("valid"), batch_size=128, shuffle=True
)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                4,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                8, 4, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                4, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 188, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.up(self.down(x))

    def predict(self, x):
        emb = torch.flatten(self.down(x), 1)
        return self.fc(emb)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    n_chunks = len(dataloader)
    for i, (X, y) in enumerate(dataloader):
        loss = loss_fn(model, X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")


def test(dataloader, model, autoencoder_loss_fn, prediction_loss_fn):
    model.eval()
    test_autoencoder_loss = 0
    test_prediction_loss = 0
    n_chunks = len(dataloader)
    for i, (X, y) in enumerate(dataloader):
        test_autoencoder_loss += autoencoder_loss_fn(model.forward(X), X)
        test_prediction_loss += prediction_loss_fn(model.predict(X), y)
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")
    mean_autoencoder_loss = test_autoencoder_loss / n_chunks
    mean_prediction_loss = test_prediction_loss / n_chunks
    print(f"{mean_autoencoder_loss=:0.5f}, {mean_prediction_loss=:0.5f}")
    return mean_autoencoder_loss, mean_prediction_loss


def train_test_loop(ae, train_loader, valid_loader, plotter, path, epochs=100):
    if os.path.exists(path):
        raise ValueError(f"path {path} already exists!")
    autoencoder_loss_fn = nn.MSELoss()
    prediction_loss_fn = nn.CrossEntropyLoss()
    lambda_ = 0.1
    loss_fn = lambda model, X, y: (1 - lambda_) * autoencoder_loss_fn(
        model.forward(X), X
    ) + (lambda_ * prediction_loss_fn(model.predict(X), y) if lambda_ != 0 else 0)

    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)  # ???
    chunk = next(iter(valid_loader))[0]
    chunk_det = chunk.detach().numpy()
    try:
        for epoch in range(epochs):
            if epoch == 10:
                lambda_ = 0.5
            if epoch == 20:
                print("Changing to prediction mode!")
                for p in ae.down.parameters():
                    p.requires_grad = False
                lambda_ = 1.0
            print(f"{epoch=}, {lambda_=}")
            train(train_loader, ae, loss_fn, optimizer)
            mean_autoencoder_loss, mean_prediction_loss = test(
                valid_loader, ae, autoencoder_loss_fn, prediction_loss_fn
            )
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
            fig.suptitle(
                f"epoch={epoch}, {mean_autoencoder_loss=:0.5f}, {mean_prediction_loss=:0.5f}"
            )
            plt.savefig(f"training_output/{epoch:04d}.png")
            plt.close()
    except KeyboardInterrupt:
        pass
    torch.save(ae, path)
