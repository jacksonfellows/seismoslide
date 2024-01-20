# Plot the saved outputs of a model.id

import glob
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import wandb
from train import CLASSES, find_TP_FP_FN


def plot_results(X, y, y_pred):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey="row", layout="tight")
    axs[0].plot(X)
    for classi, (classl, color) in enumerate(
        zip(CLASSES, ["blue", "red", "yellow", "green"])
    ):
        axs[1].plot(
            y[classi],
            label=f"expected {classl}",
            color=color,
        )
        axs[1].plot(
            y_pred[classi],
            label=f"output {classl}",
            color=color,
            linestyle="dashed",
        )
    plt.legend()
    plt.show()
    plt.close(fig)  # ?


def compute_stats(dirpath, threshold=0.5):
    dirpath = Path(dirpath)
    with open(dirpath / "config.json", "r") as f:
        config = json.load(f)
    config["threshold"] = threshold
    wandb.config = config
    TP, FP, FN = np.zeros(3), np.zeros(3), np.zeros(3)
    for path in glob.glob(str(dirpath / f"*.npz")):
        print(f"loading stats from {path}")
        npz = np.load(path)
        for batchi in range(npz["y"].shape[0]):
            tp, fp, fn = find_TP_FP_FN(npz["y"][batchi], npz["y_pred"][batchi])
            TP += tp
            FP += fp
            FN += fn
        print(TP, FP, FN)
    return TP, FP, FN


def compute_stuff(stats):
    TP, FP, FN = stats
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1


def compute_roc_dumb(dirpath):
    thresholds = np.arange(0, 1, 0.05)
    P, R, F1 = (
        np.zeros((len(thresholds), len(CLASSES))),
        np.zeros((len(thresholds), len(CLASSES))),
        np.zeros((len(thresholds), len(CLASSES))),
    )
    for i, threshold in enumerate(thresholds):
        p, r, f1 = compute_stuff(compute_stats(dirpath, threshold))
        P[i] = p
        R[i] = r
        F1[i] = f1
    return thresholds, P, R, F1


def plot_roc_stats(roc_stats):
    thresholds, P, R, F1 = roc_stats
    fig, axs = plt.subplots(nrows=1, ncols=3, layout="tight", sharey=True)
    for classi in range(len(CLASSES)):
        axs[classi].set_title(CLASSES[classi])
        axs[classi].plot(thresholds, P[:, classi], label="Precision")
        axs[classi].plot(thresholds, R[:, classi], label="Recall")
        axs[classi].plot(thresholds, F1[:, classi], label="F1")
    axs[0].legend()
    plt.show()
