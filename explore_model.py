import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.functional as F
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import train
import wandb


def plot_results(X, y, y_pred):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey="row", layout="tight")
    axs[0].plot(X)
    for classi, (classl, color) in enumerate(
        zip(train.CLASSES, ["blue", "red", "yellow", "green"])
    ):
        axs[1].plot(
            y[classi].repeat(wandb.config["stride"]),
            label=f"expected {classl}",
            color=color,
        )
        axs[1].plot(
            y_pred[classi].repeat(wandb.config["stride"]),
            label=f"output {classl}",
            color=color,
            linestyle="dashed",
        )
    plt.show()
    plt.close(fig)  # ?


api = wandb.Api()


def load_run(run_path):
    run = api.run(run_path)
    model_artifact = [a for a in run.logged_artifacts() if a.type == "model"][0]
    with tempfile.TemporaryDirectory() as tmpdir:
        model_artifact.download(tmpdir)
        model_path = os.listdir(tmpdir)[0]
        model = torch.load(Path(tmpdir) / model_path)
    return run.config, model


def plot(config, model):
    wandb.config = config
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset),
        wandb.config["batch_size"],
        shuffle=True,
    )
    wandb.config["threshold"] = 0.5
    d = next(iter(valid_loader))
    X, y = d["X"], d["y"]
    y_pred = model(X).detach()
    X, y = X.detach().numpy(), y.detach().numpy()
    for batchi in range(X.shape[0]):
        tp, fp, fn = train.find_TP_FP_FN(y[batchi], y_pred[batchi])
        if fp.sum() != 0 or fn.sum() != 0:
            plot_results(X[batchi][0], y[batchi], y_pred[batchi])
            # return
