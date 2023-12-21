import itertools

import numpy as np
import scipy
import seisbench.data
from matplotlib import pyplot as plt
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta

from create_dataset import normalize

# from envelopes import envelope

rng = np.random.default_rng(123)

ds = seisbench.data.PNWExotic()
# ds = seisbench.data.PNW()
ds_noise = seisbench.data.PNWNoise()
ds_i = np.arange(len(ds))[:1000]
ds_waveforms = [normalize(ds.get_waveforms(i)[0]) for i in ds_i]
ds_noise_waveforms = [normalize(ds_noise.get_waveforms(i)[0]) for i in ds_i]
ds_p_arrival_sample = [ds.metadata.iloc[i]["trace_P_arrival_sample"] for i in ds_i]


def plot_waveform(x):
    sampling_rate = 100
    t = np.arange(len(x)) / sampling_rate
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.show()


def pick(x, s, l, start, end):
    y = classic_sta_lta(x, s, l)
    # slow and dumb
    out = np.empty_like(y, dtype="bool")
    on = False
    for i in range(len(y)):
        if on:
            if y[i] < end:
                on = False
        else:
            if y[i] > start:
                on = True
        out[i] = on
    return out


def onsets(picks):
    w = np.lib.stride_tricks.sliding_window_view(picks, 2)
    # Ignores onset at index 0.
    return np.arange(1, len(picks))[(w[:, 0] == 0) & (w[:, 1] == 1)]


def score_picker(kwargs):
    score = 0
    for x, p_arrival_sample in zip(ds_waveforms, ds_p_arrival_sample):
        picked_samples = onsets(pick(x, **kwargs))
        if np.isnan(p_arrival_sample):
            continue
        if len(picked_samples) > 0:
            score += np.min(np.abs(picked_samples - p_arrival_sample))
        else:
            score += len(x) // 2
    return score / len(ds_waveforms)


def score_picker_noise(kwargs):
    score = 0
    for x in ds_noise_waveforms:
        picked_samples = onsets(pick(x, **kwargs))
        score += len(picked_samples)
    return score / len(ds_waveforms)


def lookat(picker_kwargs):
    N = 10
    fig, axs = plt.subplots(nrows=2, ncols=N, sharex=True, sharey="row")
    for i in range(N):
        x = ds_waveforms[i]
        picks = pick(x, **picker_kwargs)
        axs[0, i].plot(x)
        axs[1, i].plot(picks)
    plt.show()


def lookat_noise(picker_kwargs):
    N = 10
    fig, axs = plt.subplots(nrows=2, ncols=N, sharex=True, sharey="row")
    for i in range(N):
        x = ds_noise_waveforms[i]
        picks = pick(x, **picker_kwargs)
        axs[0, i].plot(x)
        axs[1, i].plot(picks)
    plt.show()


def grid_search(score_f):
    best_score = float("inf")
    for s, l, start, end in itertools.product(
        [100, 150, 200, 500, 1000],
        [500, 1000, 2500, 5000, 7500],
        [1, 2, 5, 10],
        [0.5, 1, 2],
    ):
        kwargs = dict(s=s, l=l, start=start, end=end)
        score = score_f(kwargs)
        if score < best_score:
            print(score, kwargs)
            best_score = score
