import math
import sys
import time

import torch

import wandb
from train_lppn import train_lppn_model

sweep_config = {
    "method": "random",
    "name": "sweep2",
    "metric": {"name": "valid_epoch/mean_F1", "goal": "maximize"},
    "parameters": {
        "stride": {"values": [8, 16, 32, 64, 128]},
        "base": {"values": [4, 8, 16, 32, 64]},
        "sigma": {"min": 20, "max": 120},
        "window_len": {"values": [1536, 3072, 6144]},
        "lr": {
            "distribution": "log_uniform_values",
            "max": 0.01,
            "min": 0.0001,
        },
    },
}


def do_sweep():
    # Register this function to run sweep.
    default_config = {
        "min_distance": 100,
        "window_low": 0,
        "epochs": 10,
        "batch_size": 64,
        "threshold": 0.7,
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
