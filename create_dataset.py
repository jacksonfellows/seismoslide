from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import seisbench.data
from obspy import UTCDateTime
from obspy.signal.filter import bandpass

from mydataset import MyDatasetWriter


def shift_P_arrival(
    waveform,
    trace_P_arrival_sample,
    sampling_rate,
    waveform_len_samples,
    pre_arrival_len_samples,
):
    """Shifts a waveform so that the P arrival is in a consistent
    place. Returns (shifted waveform, # of samples shifted)."""
    start_samples = int(trace_P_arrival_sample - pre_arrival_len_samples)
    return (
        waveform[start_samples : (start_samples + waveform_len_samples)],
        start_samples,
    )


pnw_exotic = seisbench.data.PNWExotic()
pnw_noise = seisbench.data.PNWNoise()
pnw = seisbench.data.PNW()

np_gen = np.random.default_rng(123)


def save_event(waveform, metadata, writer, sampling_rate):
    WAVEFORM_LEN = 6000
    PRE_ARRIVAL_LEN_SAMPLES = 3000
    waveform = waveform[0]  # Z component.
    if metadata.source_type != "noise":
        trace_P_arrival_sample = metadata.trace_P_arrival_sample
        if np.isnan(trace_P_arrival_sample):
            print(f"skipping {metadata.source_type} - no trace_P_arrival_sample")
            return
        shifted, shift_samples = shift_P_arrival(
            waveform,
            trace_P_arrival_sample,
            sampling_rate=sampling_rate,
            waveform_len_samples=WAVEFORM_LEN,
            pre_arrival_len_samples=PRE_ARRIVAL_LEN_SAMPLES,
        )
        trace_start_time = (
            UTCDateTime(metadata.trace_start_time) + shift_samples / sampling_rate
        )
    else:
        shifted = waveform[:WAVEFORM_LEN]
        trace_start_time = UTCDateTime(metadata.trace_start_time)
    if shifted.shape != (WAVEFORM_LEN,):
        print(f"skipping {metadata.source_type} - malformed shape {shifted.shape}")
        return
    writer.write_sample(
        shifted.astype("float32"),
        {
            "source_type": metadata.source_type,
            "event_id": metadata.get("event_id"),  # Noise waveforms don't have ids.
            "station_network_code": metadata.station_network_code,
            "station_code": metadata.station_code,
            "station_location_code": metadata.station_location_code,
            "trace_start_time": trace_start_time,
            "trace_P_arrival_sample": PRE_ARRIVAL_LEN_SAMPLES
            if metadata.source_type != "noise"
            else None,
        },
    )


def write_split(i_su, i_eq, i_ex, i_noise, base_path):
    with MyDatasetWriter(
        base_path,
        [
            "source_type",
            "event_id",
            "station_network_code",
            "station_code",
            "station_location_code",
            "trace_start_time",
            "trace_P_arrival_sample",
        ],
    ) as writer:
        for indices, dataset in zip(
            [i_su, i_eq, i_ex, i_noise], [pnw_exotic, pnw, pnw, pnw_noise]
        ):
            for i in indices:
                save_event(
                    dataset.get_waveforms(i),
                    dataset.metadata.loc[i],
                    writer,
                    sampling_rate=100,
                )


def make_splits(split_dir):
    all_su = pnw_exotic.metadata[pnw_exotic.metadata.source_type == "surface event"]
    all_eq = pnw.metadata[pnw.metadata.source_type == "earthquake"]
    all_ex = pnw.metadata[pnw.metadata.source_type == "explosion"]
    all_noise = pnw_noise.metadata

    # Surface event indices: Random permutation of all events.
    i_su = np_gen.permutation(all_su.index.to_numpy())

    # Take as many earthquakes, explosions, and noise as there are surface events.
    i_eq = np_gen.choice(all_eq.index.to_numpy(), size=len(i_su))
    i_ex = np_gen.choice(all_ex.index.to_numpy(), size=len(i_su))
    i_noise = np_gen.choice(all_noise.index.to_numpy(), size=len(i_su))

    assert i_su.shape == i_eq.shape == i_ex.shape == i_noise.shape

    # Split percentages:
    train_p, valid_p, test_p = 0.6, 0.3, 0.1
    assert train_p * 100 + valid_p * 100 + test_p * 100 == 100
    train_i_su, valid_i_su, test_i_su = np.array_split(
        i_su, [int(train_p * len(i_su)), int((train_p + valid_p) * len(i_su))]
    )
    train_i_eq, valid_i_eq, test_i_eq = np.array_split(
        i_eq, [int(train_p * len(i_eq)), int((train_p + valid_p) * len(i_eq))]
    )
    train_i_ex, valid_i_ex, test_i_ex = np.array_split(
        i_ex, [int(train_p * len(i_ex)), int((train_p + valid_p) * len(i_ex))]
    )
    train_i_noise, valid_i_noise, test_i_noise = np.array_split(
        i_noise, [int(train_p * len(i_noise)), int((train_p + valid_p) * len(i_noise))]
    )

    write_split(train_i_su, train_i_eq, train_i_ex, train_i_noise, split_dir / "train")
    write_split(valid_i_su, valid_i_eq, valid_i_ex, valid_i_noise, split_dir / "valid")
    write_split(test_i_su, test_i_eq, test_i_ex, test_i_noise, split_dir / "test")


def create_splits():
    split_dir = Path("./pnw_splits")
    if split_dir.exists():
        raise ValueError(f"directory {split_dir} already exists")
    split_dir.mkdir()
    make_splits(split_dir)


if __name__ == "__main__":
    create_splits()
