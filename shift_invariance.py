import pickle
from functools import cache

import numpy as np
import scipy
import seisbench.data
import torch
from matplotlib import pyplot as plt

from normalize import normalize
from train import CLASSES

models = {}
for path in [
    "model1.pickle",
    "phasenet_blurpool_3.pickle",
    "phasenet_blurpool_8.pickle",
    "phasenet_blurpool_9.pickle",
]:
    with open(path, "rb") as f:
        models[path] = pickle.load(f)

# pnw_exotic = seisbench.data.PNWExotic(component_order="Z")
valid_dataset = seisbench.data.WaveformDataset("pnw_all_2", component_order="Z").dev()


def find_peaks(y):
    find_peaks_kwargs = dict(height=0.5, distance=250)
    results = dict()
    for classi, cls in enumerate(CLASSES):
        peaks, _ = scipy.signal.find_peaks(y[classi], **find_peaks_kwargs)
        if len(peaks) > 0:
            results[cls] = peaks
    return results


def apply_model(model_path, X):
    # Note: Not batched.
    model = models[model_path]
    X = X[None, ...]
    with torch.no_grad():
        X = torch.tensor(normalize(X), dtype=torch.float32)
        y_ = model(X).numpy()
        return y_[0]


def plot(X, y):
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(X.T, "k", linewidth=0.75)
    for i, clss in enumerate(CLASSES + ["noise"]):
        axs[1].plot(y[i], label=clss)
    plt.legend()
    plt.show()


def plot_offset(model_path, i, offset):
    sample_X, sample_m = valid_dataset.get_sample(i)
    print(sample_X.shape)
    print(sample_m)
    arrival_sample = int(sample_m["trace_P_arrival_sample"])
    start = arrival_sample - offset
    X = sample_X[:, start : start + 6000]
    y = apply_model(model_path, X)
    plot(X, y)


@cache
def calc_stats(model_path, i):
    sample_X, sample_m = valid_dataset.get_sample(i)
    print(sample_X.shape)
    print(sample_m)
    arrival_sample = int(sample_m["trace_P_arrival_sample"])
    offsets = np.arange(6000)
    probs, pick_errs, peak_probs = np.zeros(6000), np.zeros(6000), np.zeros(6000)
    for i in range(len(offsets)):
        offset = offsets[i]
        start = arrival_sample - offset
        X = sample_X[:, start : start + 6000]
        y = apply_model(model_path, X)
        # plot(X, y)
        # peaks = find_peaks(y)
        prob = y[CLASSES.index(sample_m["source_type"])][offset]
        max_i = y[CLASSES.index(sample_m["source_type"])].argmax()
        peak_prob = y[CLASSES.index(sample_m["source_type"])][max_i]
        probs[i] = prob
        pick_errs[i] = offset - max_i
        peak_probs[i] = peak_prob
    return probs, pick_errs, peak_probs


# def check_avg(N, *model_paths):
#     plt.title(f"Mean of {N} Events")
#     plt.xlabel("Arrival Sample")
#     plt.ylabel("Prob. of Correct Class @ Arrival Sample")
#     plt.ylim(0, 1)
#     plt.vlines(200, 0, 1, "r")
#     plt.vlines(1500, 0, 1, "r")
#     offsets = np.arange(6000)
#     prob_means = np.zeros((len(model_paths), 6000))
#     for i in range(N):
#         if valid_dataset.metadata.iloc[i]["source_type"] != "surface event":
#             print(f"skipping sample {i}")
#             continue
#         for model_path_i, model_path in enumerate(model_paths):
#             _, probs = calc_offsets_probs(model_path, i)
#             prob_means[model_path_i] += probs
#     for model_path_i, model_path in enumerate(model_paths):
#         plt.plot(offsets, prob_means[model_path_i] / N, label=model_path)
#     plt.legend()
#     plt.show()


def check(i, *model_paths):
    fig, axs = plt.subplots(nrows=3, sharex=True)
    axs[0].set_title(f"Validation Event #{i}")
    axs[0].vlines([200, 1500], 0, 1, "r")
    axs[1].vlines([200, 1500], 0, 6000, "r")
    axs[1].set_ylim(0, 200)
    axs[2].vlines([200, 1500], 0, 1, "r")
    axs[0].set_ylabel("Prob. @ Ref. Arrival")
    axs[1].set_ylabel("Distance of Peak from Ref. Arrival")
    axs[2].set_ylabel("Max Prob.")
    offsets = np.arange(6000)
    for model_path in model_paths:
        probs, pick_errs, peak_probs = calc_stats(model_path, i)
        axs[0].plot(offsets, probs, label=model_path)
        axs[1].plot(offsets, np.abs(pick_errs), label=model_path)
        axs[2].plot(offsets, peak_probs, label=model_path)
    axs[0].legend()
    plt.show()


def check2(*model_paths):
    plt.xlabel("Arrival Time [Samples]")
    plt.ylabel("Pick Residual [Samples]")
    offsets = np.arange(6000)[200:1500]
    for model_path in model_paths:
        for i in range(10):
            _, pick_errs, _ = calc_stats(model_path, i)
            e = pick_errs[200:1500]
            e -= e.mean()
            (line,) = plt.plot(
                e,
                color={"model1.pickle": "red", "phasenet_blurpool_9.pickle": "blue"}[
                    model_path
                ],
                alpha=0.2,
            )
            if i == 0:
                line.set_label(model_path)
    plt.legend()
    plt.show()
