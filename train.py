import numpy as np
import scipy
import seisbench.data
import seisbench.generate as sbg
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import wandb
from normalize import normalize

dataset = seisbench.data.WaveformDataset("./pnw_all")
train_dataset = dataset.train()
valid_dataset = dataset.dev()

CLASSES = ["earthquake", "explosion", "surface event"]


def make_proba_pick(onset):
    N = wandb.config["window_len"] // wandb.config["stride"]
    x = np.arange(N) * wandb.config["stride"]
    if wandb.config["pick_label_type"] == "Gaussian":
        return np.exp(-((x - onset) ** 2) / (2 * wandb.config["sigma"] ** 2))
    if wandb.config["pick_label_type"] == "triangular":
        return np.clip(1 - np.abs(x - onset) / wandb.config["sigma"], 0, 1)
    raise ValueError(f"unsupported pick_label_type {wandb.config['pick_label_type']}")


def add_classif_output(state_dict):
    waveform, metadata = state_dict["X"]
    P_arrival_sample = metadata.get("trace_P_arrival_sample")
    assert len(waveform) % wandb.config["stride"] == 0
    N = len(waveform) // wandb.config["stride"]
    probs = np.zeros((len(CLASSES), N), dtype="float32")
    if metadata["source_type"] == "noise":
        pass
    else:
        classi = CLASSES.index(metadata["source_type"])
        # Round onset to nearest bin.
        onset = wandb.config["stride"] * (P_arrival_sample // wandb.config["stride"])
        probs[classi] = make_proba_pick(onset)
    state_dict["y"] = (probs, None)  # Need to indicate empty metadata!


def my_normalize(state_dict):
    waveform, metadata = state_dict["X"]
    state_dict["X"] = normalize(waveform).astype("float32"), metadata


def add_channel_dim(state_dict):
    waveform, metadata = state_dict["X"]
    state_dict["X"] = waveform[None, ...], metadata


def add_metadata_i(state_dict):
    waveform, metadata = state_dict["X"]
    state_dict["metadata_i"] = metadata["metadata_i"], None


def make_generator(dataset):
    gen = sbg.GenericGenerator(dataset)
    gen.augmentation(
        sbg.RandomWindow(
            windowlen=wandb.config["window_len"],
            low=wandb.config["window_low"],
            high=wandb.config["window_high"],
        )
    )
    # TODO: Explore normalization options.
    # gen.augmentation(
    #     sbg.Normalize(
    #         demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="std"
    #     )
    # )
    gen.augmentation(my_normalize)
    gen.augmentation(add_classif_output)
    if wandb.config.get("add_channel_dim"):
        gen.augmentation(add_channel_dim)
    gen.augmentation(add_metadata_i)
    return gen


def find_TP_FP_FN(y_true, y_pred):
    # y_true.shape == y_pred.shape == (num_classes, num_segments)
    TP, FP, FN = np.zeros(3), np.zeros(3), np.zeros(3)
    for classi in range(len(CLASSES)):
        peaks_true, _ = scipy.signal.find_peaks(
            y_true[classi],
            height=wandb.config["threshold"],
            prominence=wandb.config["threshold"] / 2,
        )
        peaks_pred, _ = scipy.signal.find_peaks(
            y_pred[classi],
            height=wandb.config["threshold"],
            prominence=wandb.config["threshold"] / 2,
        )
        peaks_true, peaks_pred = (
            wandb.config["stride"] * peaks_true,
            wandb.config["stride"] * peaks_pred,
        )
        if len(peaks_true) == 0:
            FP[classi] += len(peaks_pred)
        else:
            assert len(peaks_true) == 1
            if len(peaks_pred) == 0:
                FN[classi] += 1
            else:
                close = np.sum(
                    np.abs(peaks_pred - peaks_true[0]) < wandb.config["min_distance"]
                )
                TP[classi] += close
                FP[classi] += len(peaks_pred) - close
    return TP, FP, FN


class MetricLogger:
    def __init__(self, split):
        self.split = split
        self.TP, self.FP, self.FN = np.zeros(3), np.zeros(3), np.zeros(3)
        self.total_loss = 0
        self.n_batches = 0

    def update_batch(self, y, y_pred, loss):
        self.total_loss += loss
        for batchi in range(y.shape[0]):
            tp, fp, fn = find_TP_FP_FN(y[batchi], y_pred[batchi])
            self.TP += tp
            self.FP += fp
            self.FN += fn
        self.n_batches += 1

    def do_log(self):
        if self.n_batches == 0:
            return
        mean_loss = self.total_loss / self.n_batches
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = self.TP / (self.TP + self.FP)
            recall = self.TP / (self.TP + self.FN)
            f1 = 2 / (1 / precision + 1 / recall)
        log = {f"{self.split}/mean_loss": mean_loss}
        for classi, classl in enumerate(CLASSES):
            log[f"{self.split}/{classl}_precision"] = precision[classi]
            log[f"{self.split}/{classl}_recall"] = recall[classi]
            log[f"{self.split}/{classl}_F1"] = f1[classi]
        log[f"{self.split}/mean_F1"] = np.mean(f1)
        wandb.log(log)
        self.total_loss = 0
        self.TP[:] = self.FP[:] = self.FN[:] = 0
        self.n_batches = 0


def do_loop(dataloader, model, loss_fn, optimizer, do_train):
    if do_train:
        model.train()
    else:
        model.eval()
    n_batches = len(dataloader)
    split = "train" if do_train else "valid"
    batch_logger = MetricLogger(split)
    epoch_logger = MetricLogger(f"{split}_epoch")
    for i, d in enumerate(dataloader):
        if i % 10 == 0:
            print(f"{i}/{n_batches}")
            batch_logger.do_log()
        y = d["y"]
        y_pred = model(d["X"])
        loss = loss_fn(y_pred, y)
        if do_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        y_pred_detached = y_pred.detach()
        loss_detached = loss.detach()
        batch_logger.update_batch(y, y_pred_detached, loss_detached)
        epoch_logger.update_batch(y, y_pred_detached, loss_detached)
    epoch_logger.do_log()


def train_test_loop(model, train_loader, valid_loader, path, epochs):
    wandb.watch(model, log_freq=100)
    wandb.config["optimizer"] = "Adam"
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
    loss_fn = torch.nn.BCELoss()

    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            do_loop(train_loader, model, loss_fn, optimizer, do_train=True)
            do_loop(valid_loader, model, loss_fn, None, do_train=False)
    except KeyboardInterrupt:
        pass
    torch.save(model, path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    wandb.finish()
