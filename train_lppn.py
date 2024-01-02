import argparse

from torch.utils.data import DataLoader

import lppn_model
import train
import wandb


def train_lppn_model():
    model = lppn_model.Model(
        n_stride=wandb.config["stride"], n_channel=wandb.config["base"]
    )
    train_loader = DataLoader(
        train.make_generator(train.train_dataset, wandb.config),
        wandb.config["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset, wandb.config),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int)
    parser.add_argument("--base", type=int)
    parser.add_argument("--path")
    parser.add_argument("--sigma", type=int)
    parser.add_argument("--window_len", type=int)
    parser.add_argument("--window_low", type=int)
    parser.add_argument("--window_high", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threshold", type=float)
    args = vars(parser.parse_args())
    wandb.init(
        project="seismoslide",
        config=dict(
            model="LPPN",
            **args,
            batch_size=64,
            min_distance=100,
        ),
    )
    train_lppn_model()
