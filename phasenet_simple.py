import sys
import time

import torch
from torch.utils.data import DataLoader

import my_phasenet
import train
import wandb

config = {
    "window_len": 3000,
    "lr": 1e-3,
    "pick_label_type": "Gaussian",
    "sigma": 100,
    "epochs": 25,
    "min_distance": 100,
    "window_low": 0,
    "batch_size": 64,
    "threshold": 0.5,
    "model": "Seisbench Phasenet 1",
    "stride": 1,
    "add_channel_dim": False,
}


def run():
    with wandb.init(config=config):
        # Set window_high based on window_len.
        wandb.config["window_high"] = wandb.config["window_len"] + 1500 - 200
        # Set a unique path.
        wandb.config["path"] = f"sweep_{int(time.time())}.pt"
        torch.set_num_threads(1)

        model = my_phasenet.PhaseNet(
            in_channels=1,
            classes=3,
            phases=["earthquake", "explosion", "surface event"],  # class names,
            sampling_rate=100,
        )

        train_loader = DataLoader(
            train.make_generator(train.train_dataset),
            wandb.config["batch_size"],
            shuffle=True,
        )
        valid_loader = DataLoader(
            train.make_generator(train.valid_dataset),
            wandb.config["batch_size"],
            shuffle=True,
        )
        train.train_test_loop(
            model,
            train_loader,
            valid_loader,
            wandb.config["path"],
            wandb.config["epochs"],
        )


if __name__ == "__main__":
    run()
