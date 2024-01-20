import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.functional as F
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
        )
