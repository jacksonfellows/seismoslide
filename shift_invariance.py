import pickle

import scipy
import seisbench.data
import torch
from matplotlib import pyplot as plt

from normalize import normalize
from train import CLASSES

with open("model1.pickle", "rb") as f:
    model = pickle.load(f)

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


def apply_model(X):
    # Note: Not batched.
    X = X[None, ...]
    with torch.no_grad():
        X = torch.tensor(normalize(X), dtype=torch.float32)
        y_ = model(X).numpy()
        return y_[0]


def plot(X, y):
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(X.T)
    for i, clss in enumerate(CLASSES + ["noise"]):
        axs[1].plot(y[i], label=clss)
    plt.legend()
    plt.show()


def check(i):
    sample_X, sample_m = valid_dataset.get_sample(i)
    print(sample_X.shape)
    print(sample_m)
    arrival_sample = int(sample_m["trace_P_arrival_sample"])
    offsets = list(range(0, 6000, 1))
    probs = []
    for offset in offsets:
        start = arrival_sample - offset
        X = sample_X[:, start : start + 6000]
        y = apply_model(X)
        # plot(X, y)
        # peaks = find_peaks(y)
        prob = y[CLASSES.index(sample_m["source_type"])][offset]
        probs.append(prob)
    plt.xlabel("Arrival Sample")
    plt.ylabel("Prob. of Correct Class @ Arrival Sample")
    plt.plot(offsets, probs)
    plt.show()
