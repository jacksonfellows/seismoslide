import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import train
import wandb

api = wandb.Api()


def load_run(run_path):
    run = api.run(run_path)
    model_artifact = [a for a in run.logged_artifacts() if a.type == "model"][0]
    with tempfile.TemporaryDirectory() as tmpdir:
        model_artifact.download(tmpdir)
        model_path = os.listdir(tmpdir)[0]
        model = torch.load(Path(tmpdir) / model_path)
    return run.config, model


def apply_to_valid(config, model):
    wandb.config = config
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset),
        wandb.config["batch_size"],
        shuffle=True,
    )
    dirpath = Path("valid_" + Path(config["path"]).stem)
    if dirpath.exists():
        raise ValueError(f"Directory {dirpath} already exists!")
    dirpath.mkdir()
    # Save config.
    with open(dirpath / "config.json", "w") as f:
        json.dump(config, f)
    n_batches = len(valid_loader)
    for i, d in enumerate(valid_loader):
        batchpath = dirpath / f"{i}.npz"
        if i % 10 == 0:
            print(f"{i}/{n_batches}")
        y = d["y"]
        X = d["X"]
        y_pred = model(X)
        np.savez_compressed(
            batchpath,
            y=y.detach().numpy(),
            X=X.detach().numpy(),
            y_pred=y_pred.detach().numpy(),
            metadata_i=d["metadata_i"],
        )


def plot_emb_phasenet(config, model):
    wandb.config = config
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset),
        64 * 1000,  # Include all samples in one batch.
        shuffle=True,
    )
    d = next(iter(valid_loader))
    x = d["X"]
    batchlen = x.shape[0]
    model.eval()
    x = model.activation(model.in_bn(model.inc(x)))

    skips = []
    for i, (conv_same, bn1, conv_down, bn2) in enumerate(model.down_branch):
        x = model.activation(bn1(conv_same(x)))

        if conv_down is not None:
            skips.append(x)
            if i == 1:
                x = F.pad(x, (2, 3), "constant", 0)
            elif i == 2:
                x = F.pad(x, (1, 3), "constant", 0)
            elif i == 3:
                x = F.pad(x, (2, 3), "constant", 0)

            x = model.activation(bn2(conv_down(x)))

    skips.append(x)  # Not a real skip.

    # Could plot the skip layers but the deepest layer seems to work best.

    emb_np = skips[-1].reshape(batchlen, -1).detach().numpy()
    # xy = PCA(n_components=2).fit_transform(emb_np)
    xy = TSNE(n_components=2).fit_transform(emb_np)
    class_colors = {
        "noise": "grey",
        "earthquake": "red",
        "explosion": "green",
        "surface event": "yellow",
    }
    for cls, clr in class_colors.items():
        # d["source_type"] is a list for some reason.
        I = [c == cls for c in d["source_type"]]
        plt.scatter(xy[I, 0], xy[I, 1], c=clr, alpha=0.5, label=cls)
    plt.legend()
    plt.show()
