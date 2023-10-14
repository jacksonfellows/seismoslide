import itertools

import pandas as pd


def q(m):
    return (
        m["event_id"]
        + "."
        + m["station_network_code"]
        + "."
        + m["station_code"]
        + "."
        + m["station_location_code"]
    )


def test_splits_unique():
    splits = [
        pd.read_csv(path)
        for path in [
            "pnw_splits/train/metadata.csv",
            "pnw_splits/valid/metadata.csv",
            "pnw_splits/test/metadata.csv",
        ]
    ]
    for a, b in itertools.combinations(splits, 2):
        assert len(set(q(a)) & set(q(b))) == 0
