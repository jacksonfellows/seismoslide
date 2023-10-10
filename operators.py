from collections import namedtuple
from functools import wraps

import numpy as np
import scipy
from obspy.signal.filter import bandpass

OperatorDef = namedtuple("OperatorDef", ("arity", "delta_rank", "apply", "cell_rank"))


def make_same_rank(f):
    @wraps(f)
    def wrapper(*args):
        max_rank = max(len(x.shape) for x in args)
        new_args = []
        for arg in args:
            while len(arg.shape) < max_rank:
                arg = np.expand_dims(arg, -1)
            new_args.append(arg)
        return f(*new_args)

    return wrapper


@make_same_rank
def safe_add(x, y):
    if x.shape[-1] < y.shape[-1]:
        y = y[..., : x.shape[-1]]
    elif y.shape[-1] < x.shape[-1]:
        x = x[..., : y.shape[-1]]
    return x + y


@make_same_rank
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
    "argmax": OperatorDef(
        arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.argmax(x, axis=-1)
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
    "re_fft": OperatorDef(
        arity=1,
        delta_rank=0,
        cell_rank=1,
        apply=lambda x: np.real(scipy.fft.fft(x, axis=-1)),
    ),
    # "im_fft": OperatorDef(
    #     arity=1, delta_rank=0, cell_rank=1, apply=lambda x: np.imag(scipy.fft.fft(x, axis=-1))
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
}
