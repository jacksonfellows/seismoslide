from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import seisbench.data
from obspy import UTCDateTime
from obspy.signal.filter import bandpass


def shift_P_arrival(
    waveform,
    trace_P_arrival_sample,
    sampling_rate=100,
    waveform_len_samples=6000,
    pre_arrival_len_samples=1000,
):
    """Shifts a waveform so that the P arrival is in a consistent
    place. Returns (shifted waveform, # of samples shifted)."""
    start_samples = int(trace_P_arrival_sample - pre_arrival_len_samples)
    return (
        waveform[:, start_samples : (start_samples + waveform_len_samples)],
        start_samples,
    )


def normalize(waveform):
    """Normalize a waveform for testing/prediction."""
    normalized = scipy.signal.detrend(waveform)  # Should I also remove mean?
    normalized /= np.std(normalized)  # std vs max?
    # Same frequency range as EQTransformer.
    normalized = bandpass(
        normalized, freqmin=1, freqmax=45, df=100, corners=2, zerophase=True
    )
    return normalized


def qualified_station_name(metadata):
    return (
        metadata.station_network_code
        + "."
        + metadata.station_code
        + "."
        + metadata.station_location_code
    )


pnw_exotic = seisbench.data.PNWExotic()
pnw = seisbench.data.PNW()

np_gen = np.random.default_rng(123)


def save_event(waveform, metadata, writer, sampling_rate):
    trace_P_arrival_sample = metadata.trace_P_arrival_sample
    if np.isnan(trace_P_arrival_sample):
        print("skipping event - no trace_P_arrival_sample")
        return
    shifted, shift_samples = shift_P_arrival(
        waveform, trace_P_arrival_sample, sampling_rate=sampling_rate
    )
    normalized = normalize(shifted)
    if normalized.shape != (3, 6000):
        print(f"skipping event - malformed shape {normalized.shape}")
        return
    trace_start_time = (
        UTCDateTime(metadata.trace_start_time) + shift_samples / sampling_rate
    )
    writer.add_trace(
        {
            "source_type": metadata.source_type,
            "event_id": metadata.event_id,
            "station_network_code": metadata.station_network_code,
            "station_code": metadata.station_code,
            "station_location_code": metadata.station_location_code,
            "trace_start_time": trace_start_time,
        },
        normalized,
    )


def write_split(i_eq, i_su, base_path):
    metadata_path = base_path / "metadata.csv"
    waveforms_path = base_path / "waveforms.hdf5"
    with seisbench.data.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        sampling_rate = 100
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": sampling_rate,
        }
        for indices, dataset in zip([i_eq, i_su], [pnw, pnw_exotic]):
            for i in indices:
                save_event(
                    dataset.get_waveforms(i),
                    dataset.metadata.loc[i],
                    writer,
                    sampling_rate,
                )


def make_splits(split_dir):
    all_su = pnw_exotic.metadata[pnw_exotic.metadata.source_type == "surface event"]
    all_eq = pnw.metadata[pnw.metadata.source_type == "earthquake"]

    q_eq = qualified_station_name(all_eq)
    q_su = qualified_station_name(all_su)

    # Awful.
    counts_eq = q_eq.to_frame().join(
        q_su.value_counts().reindex(q_eq.unique(), fill_value=0), on=0
    )["count"]

    p = counts_eq.to_numpy()
    p = p / p.sum()

    # Earthquake indices: Random choice weighed by # of surface events/station.
    i_eq = np_gen.choice(all_eq.index.to_numpy(), size=len(all_su), replace=False, p=p)

    # Surface event indices: Random permutation of all events.
    i_su = np_gen.permutation(all_su.index.to_numpy())

    assert i_eq.shape == i_su.shape

    # Split percentages:
    train_p, valid_p, test_p = 0.6, 0.3, 0.1
    assert train_p * 100 + valid_p * 100 + test_p * 100 == 100
    train_i_eq, valid_i_eq, test_i_eq = np.array_split(
        i_eq, [int(train_p * len(i_eq)), int((train_p + valid_p) * len(i_eq))]
    )
    train_i_su, valid_i_su, test_i_su = np.array_split(
        i_su, [int(train_p * len(i_su)), int((train_p + valid_p) * len(i_su))]
    )

    write_split(train_i_eq, train_i_su, split_dir / "train")
    write_split(valid_i_eq, valid_i_su, split_dir / "valid")
    write_split(test_i_eq, test_i_su, split_dir / "test")


def create_splits():
    split_dir = Path("./pnw_splits")
    if split_dir.exists():
        raise ValueError(f"directory {split_dir} already exists")
    split_dir.mkdir()
    make_splits(split_dir)


if __name__ == "__main__":
    create_splits()
