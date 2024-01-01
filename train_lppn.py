import argparse

from torch.utils.data import DataLoader

import lppn_model
import train
import wandb


def train_lppn_model(args):
    config = dict(
        model="LPPN",
        **args,
        batch_size=64,
        min_distance=100,
    )
    wandb.init(project="seismoslide", config=config)
    model = lppn_model.Model(n_stride=config["stride"], n_channel=config["base"])
    train_loader = DataLoader(
        train.make_generator(train.train_dataset, config),
        config["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset, config),
        config["batch_size"],
        shuffle=True,
    )
    logger = train.MetricLogger(min_distance=config["min_distance"], S=config["stride"])
    train.train_test_loop(
        model, train_loader, valid_loader, config["path"], config["epochs"], logger
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride")
    parser.add_argument("--base")
    parser.add_argument("--path")
    parser.add_argument("--sigma")
    parser.add_argument("--window_len")
    parser.add_argument("--window_low")
    parser.add_argument("--window_high")
    parser.add_argument("--epochs")
    args = parser.parse_args()
    train_lppn_model(args)
