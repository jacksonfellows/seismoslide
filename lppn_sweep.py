import math
import sys
import time

import torch

import wandb
from train_lppn import train_lppn_model

sweep_config = {
    "method": "random",
    "name": "sweep4",
    "description": "Test forms of pick labeling.",
    "metric": {"name": "valid_epoch/mean_F1", "goal": "maximize"},
    "parameters": {
        "stride": {"value": 32},
        "base": {"value": 32},
        "sigma": {"values": [25, 50, 100, 150]},
        "window_len": {"value": 6144},
        "lr": {"value": 0.001},
        "pick_label_type": {"values": ["Gaussian", "triangular"]},
    },
}


def do_sweep():
    # Register this function to run sweep.
    default_config = {
        "min_distance": 100,
        "window_low": 0,
        "epochs": 10,
        "batch_size": 64,
        "threshold": 0.5,
    }
    with wandb.init(config=default_config):
        # Set window_high based on window_len.
        wandb.config["window_high"] = wandb.config["window_len"] + 1500 - 200
        # Set a unique path.
        wandb.config["path"] = f"sweep_{int(time.time())}.pt"
        torch.set_num_threads(1)
        train_lppn_model()


if __name__ == "__main__":
    sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=do_sweep, count=4, project="seismoslide")
