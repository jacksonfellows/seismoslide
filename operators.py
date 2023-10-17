from collections import namedtuple
from functools import wraps

import numpy as np
import scipy
from obspy.signal.filter import bandpass

OperatorDef = namedtuple("OperatorDef", ("arity", "delta_rank", "apply", "cell_rank"))


def make_same_dims(f):
    @wraps(f)
    def wrapper(x, y):
        max_rank = max(len(x.shape), len(y.shape))
        # Make same rank.
        while len(x.shape) < max_rank:
            x = np.expand_dims(x, -1)
        while len(y.shape) < max_rank:
            y = np.expand_dims(y, -1)
        # Make same dims.
        min_dims = tuple(min(a, b) for a, b in zip(x.shape, y.shape))
        x = x[tuple(slice(0, i) for i in min_dims)]
        y = y[tuple(slice(0, i) for i in min_dims)]
        return f(x, y)

    return wrapper


@make_same_dims
def safe_add(x, y):
    return x + y


@make_same_dims
def safe_correlate(x, y):
    res = np.empty_like(x)
    for i in np.ndindex(res.shape[:-1]):
        res[i] = scipy.signal.correlate(x[i], y[i], mode="same")
    return res


def first_half(x):
    if x.shape[-1] == 1:
        return x
    return x[..., : x.shape[-1] // 2]


def second_half(x):
    return x[..., x.shape[-1] // 2 :]


def envelope(x):
    return np.abs(scipy.signal.hilbert(x, axis=-1))


def periodogram(x):
    return scipy.signal.periodogram(x, axis=-1)[1]


def make_bandpass_operator(freqmin, freqmax):
    return lambda x: bandpass(x, freqmin, freqmax, df=100, corners=2, zerophase=True)


all_operators = {
    "min": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.min(x, axis=-1)
    ),
    "max": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.max(x, axis=-1)
    ),
    "mean": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.mean(x, axis=-1)
    ),
    "median": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.median(x, axis=-1)
    ),
    "argmax": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.argmax(x, axis=-1)
    ),
    "argmin": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.argmin(x, axis=-1)
    ),
    "skew": OperatorDef(
        arity=1,
        delta_rank=-1,
        cell_rank=1,
        apply=lambda x: scipy.stats.skew(x, axis=-1),
    ),
    "kurtosis": OperatorDef(
        arity=1,
        delta_rank=-1,
        cell_rank=1,
        apply=lambda x: scipy.stats.kurtosis(x, axis=-1),
    ),
    "len": OperatorDef(
        arity=1,
        delta_rank=-1,
        cell_rank=1,
        apply=lambda x: np.full(x.shape[:-1], x.shape[-1]),
    ),
    "re_fft": OperatorDef(
        arity=1,
        delta_rank=0,
        cell_rank=1,
        apply=lambda x: np.real(scipy.fft.fft(x, axis=-1)),
    ),
    # "im_fft": OperatorDef(
    #     arity=1,
    #     delta_rank=0,
    #     cell_rank=1,
    #     apply=lambda x: np.imag(scipy.fft.fft(x, axis=-1)),
    # ),
    "+": OperatorDef(arity=2, delta_rank=0, cell_rank=0, apply=safe_add),
    "corr": OperatorDef(arity=2, delta_rank=0, cell_rank=1, apply=safe_correlate),
    "first_half": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=first_half),
    "second_half": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=second_half),
    "sum": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.sum(x, axis=-1)
    ),
    "envelope": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=envelope),
    "periodogram": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=periodogram),
    "bp_1_3": OperatorDef(
        arity=1, delta_rank=0, cell_rank=1, apply=make_bandpass_operator(1, 3)
    ),
    "bp_3_10": OperatorDef(
        arity=1, delta_rank=0, cell_rank=1, apply=make_bandpass_operator(3, 10)
    ),
    "bp_10_20": OperatorDef(
        arity=1, delta_rank=0, cell_rank=1, apply=make_bandpass_operator(10, 20)
    ),
    "bp_20_45": OperatorDef(
        arity=1, delta_rank=0, cell_rank=1, apply=make_bandpass_operator(20, 45)
    ),
    "num_peaks": OperatorDef(
        arity=1,
        delta_rank=-1,
        cell_rank=1,
        apply=lambda x: np.apply_along_axis(
            lambda y: scipy.signal.find_peaks(y)[0].shape[0], -1, x
        ),
    ),
    "spectrogram": OperatorDef(
        arity=1,
        delta_rank=+1,
        cell_rank=1,
        apply=lambda x: scipy.signal.spectrogram(x, axis=-1)[2],
    ),
}
