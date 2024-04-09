from pathlib import Path

import numpy as np
import scipy
import seisbench.data
import torch
from matplotlib import pyplot as plt
from obspy import UTCDateTime
from obspy.clients.filesystem.tsindex import Client

from explore_model import load_run
from normalize import normalize
from train import CLASSES

# config, model = load_run("team-jackson/seismoslide/uxm31osy")
config, model = load_run("team-jackson/seismoslide/c8dzg3c1")

client = Client("test-rover-repo/data/timeseries.sqlite")


def find_peaks(y):
    find_peaks_kwargs = dict(height=0.5, distance=250)
    results = dict()
    for classi, cls in enumerate(CLASSES):
        peaks, _ = scipy.signal.find_peaks(y[classi], **find_peaks_kwargs)
        if len(peaks) > 0:
            results[cls] = peaks
    return results


batch_size = 64
window_len_s = 60
sampling_rate = 100
window_len = sampling_rate * window_len_s


def save_event(writer, X, peaks, info, starttime):
    for cls, samples in peaks.items():
        for sample in samples:
            writer.add_trace(
                dict(
                    source_type=cls,
                    trace_P_arrival_sample=sample,
                    trace_start_time=starttime,
                    **info
                ),
                X,
            )


def apply_batch(X, writer, batch_start_time, trace_info):
    with torch.no_grad():
        X = torch.tensor(normalize(X), dtype=torch.float32)
        y_ = model(X).numpy()
        for batchi in range(X.shape[0]):
            peaks = find_peaks(y_[batchi])
            if len(peaks) > 0:
                starttime = batch_start_time + window_len_s * batchi
                save_event(writer, X[batchi].numpy(), peaks, trace_info, starttime)


def apply(tr, writer):
    assert tr.stats.sampling_rate == sampling_rate
    trace_info = dict(
        station_network_code=tr.stats.network,
        station_code=tr.stats.station,
        station_location_code=tr.stats.location,
        station_channel_code=tr.stats.channel,
    )
    X = tr.data[:-1]
    Q = 15 * sampling_rate
    X1 = X.reshape((batch_size, 1, window_len))
    X2 = X[Q : -3 * Q].reshape((batch_size - 1, 1, window_len))
    X3 = X[2 * Q : -2 * Q].reshape((batch_size - 1, 1, window_len))
    X4 = X[3 * Q : -Q].reshape((batch_size - 1, 1, window_len))
    start_time = tr.stats.starttime
    apply_batch(X1, writer, start_time, trace_info)
    apply_batch(X2, writer, start_time + 15, trace_info)
    apply_batch(X3, writer, start_time + 30, trace_info)
    apply_batch(X4, writer, start_time + 45, trace_info)


def go():
    d = Path("inference_test")
    with seisbench.data.WaveformDataWriter(
        d / "metadata.csv", d / "waveforms.hdf5"
    ) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "Z",
            "sampling_rate": 100,
        }
        start_time = UTCDateTime(2024, 1, 1, 0, 0, 0)
        for _ in range(10):
            print(start_time)
            st = client.get_waveforms(
                "*",
                "*",
                "*",
                "*",
                start_time,
                start_time + batch_size * window_len_s,
                merge=True,
            )
            for tr in st:
                print(tr)
                if len(tr) < batch_size * window_len:
                    print("skipping trace")
                    continue
                apply(tr, writer)
            start_time += batch_size * window_len_s
