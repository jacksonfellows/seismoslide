# Convert our picks to a seisbench dataset.

import sqlite3
from pathlib import Path

import numpy as np
import seisbench.data

events_dir = Path("/Users/jackson/pickhelper/all_events/events/")
picks_db_path = Path("/Users/jackson/Downloads/picks_backup_2024-03-17-17-49-39")


# Copied from server.py.
def get_picks(event_id):
    with sqlite3.connect(picks_db_path) as cur:
        res = cur.execute(
            "SELECT channel_id, pick_sample, MAX(created_time) FROM picks WHERE event_id = ? GROUP BY channel_id;",
            (event_id,),
        )
        return {row[0]: row[1] for row in res}


def get_picked_events():
    with sqlite3.connect(picks_db_path) as cur:
        res = cur.execute(
            "SELECT event_id, trace_start_time FROM events WHERE n_user_picks > 0;"
        )
        return [
            dict(event_id=row[0], trace_start_time=row[1]) for row in res.fetchall()
        ]


events = get_picked_events()


def get_trace_metadata(event_id, trace_start_time, channel_id, pick_sample):
    station_network_code, station_code, station_location_code, station_channel_code = (
        channel_id.split(".")
    )
    # This is what I saw for no location codes in other datasets.
    if station_location_code == "":
        station_location_code = "--"
    # Strip off the component.
    station_channel_code = station_channel_code[:2]
    assert len(station_channel_code) == 2

    return {
        "source_type": "surface event",
        "event_id": event_id,
        "station_network_code": station_network_code,
        "station_code": station_code,
        "station_location_code": station_location_code,
        "station_channel_code": station_channel_code,
        "trace_start_time": trace_start_time,
        "trace_P_arrival_sample": pick_sample,
    }


def get_trace_waveform(event_id, channel_id):
    X = np.load(events_dir / event_id / (channel_id + ".npy"))
    assert len(X.shape) == 1
    # Make sure trace is 2 minutes at 100 Hz.
    assert np.abs(X.shape[0] - 2 * 60 * 100) < 1 * 100
    return X[None, ...]


def write_event(writer: seisbench.data.WaveformDataWriter, event_dict):
    event_id = event_dict["event_id"]
    trace_start_time = event_dict["trace_start_time"]
    picks = get_picks(event_id)
    for channel_id, pick_sample in picks.items():
        if pick_sample is None:
            print("None pick", event_id, channel_id)
            continue
        try:
            metadata = get_trace_metadata(
                event_id, trace_start_time, channel_id, pick_sample
            )
            waveform = get_trace_waveform(event_id, channel_id)
            writer.add_trace(metadata, waveform)
        except AssertionError:
            print("bad shape", event_id, channel_id)
            pass


def write_all():
    base_path = Path("./pnw_exotic_more")
    base_path.mkdir()
    with seisbench.data.WaveformDataWriter(
        base_path / "metadata.csv", base_path / "waveforms.hdf5"
    ) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "Z",
            "sampling_rate": 100,
        }
        for event in events:
            write_event(writer, event)
