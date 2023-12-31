import argparse

from torch.utils.data import DataLoader

import lppn_model
import train
import wandb


def train_lppn_model(S, base, path, epochs):
    config = dict(
        model="LPPN",
        S=S,
        base=base,
        batch_size=64,
        min_distance=100,
        sigma=50,
    )
    sigmas = 4 * [config["sigma"]]
    wandb.init(project="seismoslide", config=config)
    model = lppn_model.Model(n_stride=S, n_channel=base)
    train_loader = DataLoader(
        train.make_generator(train.train_dataset, S, sigmas),
        config["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        train.make_generator(train.valid_dataset, S, sigmas),
        config["batch_size"],
        shuffle=True,
    )
    logger = train.MetricLogger(min_distance=config["min_distance"], S=S)
    train.train_test_loop(model, train_loader, valid_loader, path, epochs, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride")
    parser.add_argument("--base")
    parser.add_argument("--path")
    parser.add_argument("--epochs")
    args = parser.parse_args()
    train_lppn_model(
        S=int(args.stride), base=int(args.base), path=args.path, epochs=int(args.epochs)
    )
