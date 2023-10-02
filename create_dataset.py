# Combine PNW and PNWExotic into one dataset with a equal number of
# surface events and earthquakes.

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


def create_dataset():
    base_path = Path.home() / ".seisbench/datasets/seismoslide_1"
    metadata_path = base_path / "metadata.csv"
    waveforms_path = base_path / "waveforms.hdf5"
    with seisbench.data.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        sampling_rate = 100
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": sampling_rate,
        }

        def save_event(waveform, metadata):
            trace_P_arrival_sample = metadata.trace_P_arrival_sample
            if np.isnan(trace_P_arrival_sample):
                print("skipping event - no trace_P_arrival_sample")
                return
            shifted, shift_samples = shift_P_arrival(
                waveform, trace_P_arrival_sample, sampling_rate=sampling_rate
            )
            normalized = normalize(shifted)
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

        surface_events = pnw_exotic.metadata[
            pnw_exotic.metadata.source_type == "surface event"
        ]
        earthquakes = pnw.metadata[pnw.metadata.source_type == "earthquake"]
        surface_event_stations = qualified_station_name(surface_events)
        earthquake_stations = qualified_station_name(earthquakes)
        for station, count in surface_event_stations.value_counts().items():
            possible_surface_events = surface_events[surface_event_stations == station]
            possible_earthquakes = earthquakes[earthquake_stations == station]
            if len(possible_earthquakes) < count:
                count = len(possible_earthquakes)
            sampled_surface_events = possible_surface_events.sample(n=count)
            sampled_earthquakes = possible_earthquakes.sample(n=count)
            assert (
                (qualified_station_name(sampled_surface_events) == station).all()
                and (qualified_station_name(sampled_earthquakes) == station).all()
                and len(sampled_surface_events) == len(sampled_earthquakes) == count
            )
            print(
                f"sampling {count} surface events and {count} earthquakes from station {station}"
            )
            for i, row in sampled_surface_events.iterrows():
                save_event(pnw_exotic.get_waveforms(i), row)
            for i, row in sampled_earthquakes.iterrows():
                save_event(pnw.get_waveforms(i), row)
