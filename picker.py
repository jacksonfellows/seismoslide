import itertools

import numpy as np
import seisbench.data
from matplotlib import pyplot as plt
from obspy.signal.trigger import classic_sta_lta

from create_dataset import normalize
from envelopes import envelope

rng = np.random.default_rng(123)

# ds = seisbench.data.PNWExotic()
ds = seisbench.data.PNW()
ds_i = np.arange(len(ds))[:1000]
# Take envelope?
ds_normalized = [normalize(ds.get_waveforms(i)[0]) for i in ds_i]
ds_p_arrival_sample = [ds.metadata.iloc[i]["trace_P_arrival_sample"] for i in ds_i]


def plot_waveform(x):
    sampling_rate = 100
    t = np.arange(len(x)) / sampling_rate
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.show()


def sta_lta(x, s, l):
    return classic_sta_lta(x, s, l)


def pick(x, s, l, threshold):
    y = sta_lta(x, s, l)
    return y > threshold


def onsets(picks):
    w = np.lib.stride_tricks.sliding_window_view(picks, 2)
    # Ignores onset at index 0.
    return np.arange(1, len(picks))[(w[:, 0] == 0) & (w[:, 1] == 1)]


def score_picker(**kwargs):
    score = 0
    for x, p_arrival_sample in zip(ds_normalized, ds_p_arrival_sample):
        picked_samples = onsets(pick(x, **kwargs))
        if np.isnan(p_arrival_sample):
            continue
        if len(picked_samples) > 0:
            score += np.min(np.abs(picked_samples - p_arrival_sample))
        else:
            score += len(x) // 2
        score += 10 * len(picked_samples)
    return score / len(ds_normalized)


def lookat(picker_kwargs):
    N = 5
    fig, axs = plt.subplots(nrows=2, ncols=N, sharex=True)
    for axi, i in enumerate(rng.integers(0, len(ds_normalized), size=N)):
        x = ds_normalized[i]
        picks = pick(x, **picker_kwargs)
        axs[0, axi].plot(x)
        axs[1, axi].plot(picks)
    plt.show()


def grid_search():
    best_score = float("inf")
    for s, l, threshold in itertools.product(
        [5, 10, 25, 50, 75, 100],
        [500, 750, 1000, 1250, 1500],
        [1, 2.5, 5, 7.5, 10],
    ):
        kwargs = dict(s=s, l=l, threshold=threshold)
        score = score_picker(**kwargs)
        if score < best_score:
            print(score, kwargs)
            best_score = score
