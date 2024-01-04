import sys
import time

import torch
from torch.utils.data import DataLoader

import my_eqt_model
import train
import wandb

sweep_config = {
    "method": "random",
    "name": "sweep_eqt_1",
    "description": "Try out EQT.",
    "metric": {"name": "valid_epoch/mean_F1", "goal": "maximize"},
    "parameters": {
        "window_len": {"values": [3000, 6000]},
        "lr": {"value": 0.001},
        "pick_label_type": {"values": ["Gaussian", "triangular"]},
        "sigma": {"values": [20, 100, 150]},
    },
}


def do_sweep():
    default_config = {
        "min_distance": 100,
        "window_low": 0,
        "epochs": 10,
        "batch_size": 64,
        "threshold": 0.5,
        "model": "My EQT 1",
        "stride": 1,
        "add_channel_dim": True,
    }
    with wandb.init(config=default_config):
        # Set window_high based on window_len.
        wandb.config["window_high"] = wandb.config["window_len"] + 1500 - 200
        # Set a unique path.
        wandb.config["path"] = f"sweep_{int(time.time())}.pt"
        torch.set_num_threads(1)

        model = my_eqt_model.EQTransformer(
            in_channels=1,
            in_samples=wandb.config["window_len"],
            classes=3,
            phases=["earthquake", "explosion", "surface event"],
            lstm_blocks=3,
            drop_rate=0.1,
            original_compatible=False,
            norm="std",
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
