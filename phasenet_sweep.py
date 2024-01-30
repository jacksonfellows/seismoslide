import sys
import time

import torch
from torch.utils.data import DataLoader

import my_phasenet
import train
import wandb

sweep_config = {
    "method": "random",
    "name": "sweep_phasenet_no_bandpass_lr",
    "description": "Figure out best lr for PhaseNet w/o bandpass filtering.",
    "metric": {"name": "valid_epoch/mean_F1", "goal": "maximize"},
    "parameters": {
        "window_len": {"value": 3000},
        "lr": {"distribution": "log_uniform_values", "max": 0.01, "min": 0.0001},
        "pick_label_type": {"value": "Gaussian"},
        "sigma": {"values": [100, 125, 150]},
        "epochs": {"value": 25},
    },
}


def do_sweep():
    default_config = {
        "min_distance": 100,
        "window_low": 0,
        "batch_size": 64,
        "threshold": 0.5,
        "model": "Seisbench Phasenet 1",
        "stride": 1,
        "add_channel_dim": True,
    }
    with wandb.init(config=default_config):
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
    sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=do_sweep, count=1, project="seismoslide")
