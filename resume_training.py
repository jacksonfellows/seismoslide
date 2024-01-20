import sys

import torch
from torch.utils.data import DataLoader

import train
import wandb
from explore_model import load_run

if __name__ == "__main__":
    run_path = sys.argv[1]
    config, model = load_run(run_path)
    with wandb.init(config=config):
        torch.set_num_threads(1)

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
