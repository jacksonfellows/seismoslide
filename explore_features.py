from functools import cache

import numpy as np
import seisbench.data
from matplotlib import pyplot as plt

from evolve import Evolver
from operators import all_operators


@cache
def load_explore_evolver():
    dataset = seisbench.data.WaveformDataset("./pnw_splits/train")
    i = np.random.choice(np.arange(10630), 400)
    return Evolver(
        W=dataset.get_waveforms(i),
        y=(dataset.metadata.source_type == "surface event").to_numpy(dtype=int)[i],
        operators=all_operators,
    )


E = load_explore_evolver()


def y_to_label(y):
    return ["earthquake", "surface event"][y]


def y_to_color(y):
    return ["red", "blue"][y]


def plot_event(i):
    sampling_rate = 100
    W = E.W[i]
    y = E.y[i]
    x = np.arange(W.shape[1]) / sampling_rate
    for i, l in reversed(list(enumerate("ZNE"))):
        plt.plot(x, W[i], label=l)
    plt.vlines(10, W.min(), W.max(), "r", "dashed")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title(y_to_label(y))
    plt.show()


def eval_subfeatures(feature):
    yield feature, E.eval_feature_rec(feature)
    match feature:
        case (_, *x):
            for x_ in x:
                yield from eval_subfeatures(x_)


def explore_feature(feature, log=False, alpha=0.1):
    subs = list(eval_subfeatures(feature))
    fig, axs = plt.subplots(nrows=len(subs), ncols=1)
    for i, (sub, X) in enumerate(subs):
        axs[i].set_title(sub)
        if len(X.shape) == 2:
            for event_i in range(X.shape[0]):
                axs[i].plot(X[event_i], color=y_to_color(E.y[event_i]), alpha=alpha)
        elif len(X.shape) == 1:
            ii = np.argsort(X)
            if log:
                axs[i].set_yscale("symlog")
            axs[i].bar(
                np.arange(X.shape[0]), X[ii], color=[y_to_color(y) for y in E.y[ii]]
            )
        elif len(X.shape) == 3:
            # Not perfect.
            red = np.mean(X[E.y == 0], axis=0)
            blu = np.mean(X[E.y == 1], axis=0)
            axs[i].imshow(np.rot90(red), cmap="Reds", alpha=0.6)
            axs[i].imshow(np.rot90(blu), cmap="Blues", alpha=0.6)
        else:
            raise ValueError(f"rank {len(X.shape)} not supported")
    plt.show()
