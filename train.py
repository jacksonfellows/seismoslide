import numpy as np
import scipy
import seisbench.generate as sbg
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import lppn_model
import models
import wandb
from mydataset import MyDataset
from normalize import normalize

train_dataset = MyDataset("./pnw_splits/train")
valid_dataset = MyDataset("./pnw_splits/valid")

CLASSES = ["noise", "earthquake", "explosion", "surface event"]


def add_classif_output(state_dict, S, sigmas):
    waveform, metadata = state_dict["X"]
    P_arrival_sample = metadata.get("trace_P_arrival_sample")
    assert len(waveform) % S == 0
    N = len(waveform) // S
    probs = np.zeros((len(CLASSES), N), dtype="float32")
    if metadata["source_type"] == "noise":
        probs[0] = 1
    else:
        classi = CLASSES.index(metadata["source_type"])
        sigma = sigmas[classi]
        x = np.arange(N) * S
        onset = P_arrival_sample
        probs[classi] = np.exp(-((x - onset) ** 2) / (2 * sigma**2))
        probs[0] = 1 - probs[classi]
    state_dict["y"] = (probs, None)  # Need to indicate empty metadata!


def my_normalize(state_dict):
    waveform, metadata = state_dict["X"]
    state_dict["X"] = normalize(waveform).astype("float32"), metadata


WINDOW_LEN = 3072


def make_generator(dataset, S, sigmas):
    gen = sbg.GenericGenerator(dataset)
    gen.augmentation(sbg.RandomWindow(windowlen=WINDOW_LEN))
    # TODO: Explore normalization options.
    # gen.augmentation(
    #     sbg.Normalize(
    #         demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="std"
    #     )
    # )
    gen.augmentation(my_normalize)
    gen.augmentation(lambda x: add_classif_output(x, S, sigmas))
    return gen


THRESHOLD = 0.3


def find_TP_FP_FN(y_true, y_pred_logits, distance_samples, S):
    # y_true.shape == y_pred_logits.shape == (num_classes, num_segments)
    y_pred = F.softmax(y_pred_logits, dim=0)
    TP, FP, FN = np.zeros(3), np.zeros(3), np.zeros(3)
    for classi in range(1, len(CLASSES)):
        peaks_true, _ = scipy.signal.find_peaks(y_true[classi], height=THRESHOLD)
        peaks_pred, _ = scipy.signal.find_peaks(y_pred[classi], height=THRESHOLD)
        peaks_true, peaks_pred = S * peaks_true, S * peaks_pred
        if len(peaks_true) == 0:
            FP[classi - 1] += len(peaks_pred)
        else:
            assert len(peaks_true) == 1
            if len(peaks_pred) == 0:
                FN[classi - 1] += 1
            else:
                close = np.sum(np.abs(peaks_pred - peaks_true[0]) < distance_samples)
                TP[classi - 1] += close
                FP[classi - 1] += len(peaks_pred) - close
    return TP, FP, FN


class MetricLogger:
    def __init__(self, min_distance, S):
        self.min_distance = min_distance
        self.S = S
        self.TP, self.FP, self.FN = np.zeros(3), np.zeros(3), np.zeros(3)
        self.total_loss = 0
        self.n_batches = 0

    def update_batch(self, y, y_pred, loss):
        self.total_loss += loss
        for batchi in range(y.shape[0]):
            tp, fp, fn = find_TP_FP_FN(
                y[batchi], y_pred[batchi].detach(), self.min_distance, self.S
            )
            self.TP += tp
            self.FP += fp
            self.FN += fn
        self.n_batches += 1

    def do_log(self, split):
        if self.n_batches == 0:
            return
        mean_loss = self.total_loss / self.n_batches
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = self.TP / (self.TP + self.FP)
            recall = self.TP / (self.TP + self.FN)
        log = {f"{split}/mean_loss": mean_loss}
        for classi, classl in enumerate(CLASSES):
            if classi == 0:
                continue
            log[f"{split}/{classl}_precision"] = precision[classi - 1]
            log[f"{split}/{classl}_recall"] = recall[classi - 1]
        wandb.log(log)
        self.total_loss = 0
        self.TP[:] = self.FP[:] = self.FN[:] = 0
        self.n_batches = 0


def do_loop(dataloader, model, loss_fn, optimizer, logger: MetricLogger, do_train):
    if do_train:
        model.train()
    else:
        model.eval()
    n_batches = len(dataloader)
    for i, d in enumerate(dataloader):
        if i % 10 == 0:
            print(f"{i}/{n_batches}")
            logger.do_log("train" if do_train else "valid")
        y = d["y"]
        y_pred = model(d["X"])
        loss = loss_fn(y_pred, y)
        if do_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.update_batch(y, y_pred, loss)


def train_test_loop(
    model, train_loader, valid_loader, path, epochs, logger: MetricLogger
):
    wandb.watch(model, log_freq=100)
    wandb.config["optimizer"] = "Adam"
    wandb.config["lr"] = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            do_loop(train_loader, model, loss_fn, optimizer, logger, do_train=True)
            do_loop(valid_loader, model, loss_fn, None, logger, do_train=False)
    except KeyboardInterrupt:
        pass
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    wandb.finish()
    torch.save(model, path)


def plot_model(model, X, y, S):
    y = y.detach().numpy()
    y_logits = model(X)
    y_prob = F.softmax(y_logits, dim=1).detach().numpy()
    fig, axs = plt.subplots(
        nrows=2, ncols=X.shape[0], sharex=True, sharey="row", layout="tight"
    )
    for i in range(X.shape[0]):
        axs[0, i].plot(X[i])
        for classi, (classl, color) in enumerate(
            zip(CLASSES, ["blue", "red", "yellow", "green"])
        ):
            axs[1, i].plot(
                y[i, classi].repeat(S),
                label=f"expected {classl}",
                color=color,
            )
            axs[1, i].plot(
                y_prob[i, classi].repeat(S),
                label=f"output {classl}",
                color=color,
                linestyle="dashed",
            )
    # plt.legend(loc="upper left")
    plt.show()
