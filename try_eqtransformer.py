import seisbench.models
import torch
from torch.utils.data import DataLoader

import my_eqt_model
import train
import wandb


def main():
    model = my_eqt_model.EQTransformer(
        in_channels=1,
        in_samples=6000,
        classes=3,
        phases=["earthquake", "explosion", "surface event"],
        lstm_blocks=3,
        drop_rate=0.1,
        original_compatible=False,
        norm="std",
        sampling_rate=100,
    )
    wandb.init(
        project="seismoslide",
        config=dict(
            model="My EQT 01",
            batch_size=64,
            min_distance=100,
            stride=1,
            sigma=20,  # EQT paper
            pick_label_type="triangular",
            lr=0.001,
            threshold=0.5,
            epochs=10,
            window_len=6000,
            window_low=0,
            window_high=6000 + 1500 - 200,
            path="my_eqt_01.pt",
            add_channel_dim="true",
        ),
    )
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


if __name__ == "__main__":
    main()
