import sys
import time

import torch
from torch.utils.data import DataLoader

import my_phasenet
import train
import wandb

config = {
    "lr": 1e-3,
    "epochs": 50,
    "min_distance": 250,
    "window_low": 0,
    "batch_size": 64,
    "threshold": 0.5,
    "model": "Seisbench Phasenet 1",
    "add_channel_dim": False,
    "depth": 6,
    "stride": 4,
    "filters_root": 8,
    "kernel_size": 7,
    "window_len": 6000,
}

sweep_config = {
    "project": "seismoslide",
    "method": "grid",
    "name": "sweep_phasenet_params_6",
    "description": "Label size & shape.",
    "metric": {"name": "valid_epoch/surface event_F1", "goal": "maximize"},
    "parameters": {
        "pick_label_type": {"values": ["Gaussian", "triangular"]},
        "sigma": {"values": [50, 100, 150, 200]},
    },
}


def do_sweep():
    with wandb.init(config=config):
        # Set window_high based on window_len.
        wandb.config["window_high"] = wandb.config["window_len"] + 1500 - 200
        # Set a unique path.
        wandb.config["path"] = f"sweep_{int(time.time())}.pt"
        torch.set_num_threads(1)

        model = my_phasenet.PhaseNet(
            in_channels=1,
            classes=4,
            phases=[
                "earthquake",
                "explosion",
                "surface event",
                "noise",
            ],  # class names,
            sampling_rate=100,
            depth=wandb.config["depth"],
            kernel_size=wandb.config["kernel_size"],
            stride=wandb.config["stride"],
            filters_root=wandb.config["filters_root"],
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
    sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=do_sweep, count=1, project="seismoslide")
