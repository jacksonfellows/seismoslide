import pandas as pd
import seisbench.data
from matplotlib import pyplot as plt

pnw_exotic = seisbench.data.PNWExotic()
pnw = seisbench.data.PNW()


def qualified_station_name(metadata):
    return (
        metadata.station_network_code
        + "."
        + metadata.station_code
        + "."
        + metadata.station_location_code
    )


def dump_station_locs_and_counts(metadata, path):
    stations = qualified_station_name(metadata)
    columns_to_save = [
        "station_network_code",
        "station_channel_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "station_elevation_m",
    ]
    stations_info: pd.DataFrame
    stations_info = metadata.loc[stations.drop_duplicates().index][columns_to_save]
    stations_info["qualified_station"] = qualified_station_name(stations_info)
    num_traces = stations.value_counts()
    stations_info = stations_info.join(num_traces, on="qualified_station")
    stations_info.to_csv(path)


def dump_surface_events_and_earthquakes():
    dump_station_locs_and_counts(
        pnw_exotic.metadata[pnw_exotic.metadata.source_type == "surface event"],
        "surface_event_counts.csv",
    )
    dump_station_locs_and_counts(
        pnw.metadata[pnw.metadata.source_type == "earthquake"],
        "earthquake_event_counts.csv",
    )


def plot_temporal_distrib():
    pnw_exotic.metadata[
        pnw_exotic.metadata.source_type == "surface event"
    ].trace_start_time.str.slice(0, 4).astype(int).hist(bins=range(2000, 2024))
    plt.xlabel("Year")
    plt.ylabel("# of Surface Event Traces")
    plt.show()

    pnw.metadata[pnw.metadata.source_type == "earthquake"].trace_start_time.str.slice(
        0, 4
    ).astype(int).hist(bins=range(2000, 2024))
    plt.xlabel("Year")
    plt.ylabel("# of Earthquake Traces")
    plt.show()
