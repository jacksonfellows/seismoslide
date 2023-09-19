from pathlib import Path

import numpy as np
import scipy
import seisbench.data
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder

from create_dataset import qualified_station_name

# pnw_exotic = seisbench.data.PNWExotic()
# pnw = seisbench.data.PNW()

data = seisbench.data.WaveformDataset(Path.home() / ".seisbench/datasets/seismoslide_1")


def find_left_out(original_data):
    traces = (
        original_data.metadata.event_id
        + "."
        + qualified_station_name(original_data.metadata)
    )
    included_traces = (
        data.metadata.event_id + "." + qualified_station_name(data.metadata)
    )
    left_out_traces = list(set(traces) - set(included_traces))
    idx = traces.isin(left_out_traces)
    new_metadata = original_data.metadata[
        original_data.metadata.source_type.isin(["surface event", "earthquake"]) & idx
    ]
    return original_data.get_waveforms(new_metadata.index), new_metadata


def compute_features(waveforms):
    # waveforms : n_waveforms * n_channels * n_samples
    desc = scipy.stats.describe(waveforms, axis=-1)
    fft_desc = scipy.stats.describe(scipy.fft.rfft(waveforms, axis=-1), axis=-1)
    indiv_features = [
        desc.minmax[0],
        desc.minmax[1],
        desc.mean,
        desc.variance,
        desc.skewness,
        desc.kurtosis,
        fft_desc.minmax[0],
        fft_desc.minmax[1],
        fft_desc.mean,
        fft_desc.variance,
        fft_desc.skewness,
        fft_desc.kurtosis,
        np.random.rand(*waveforms.shape[:-1]),
    ]
    features = np.zeros((waveforms.shape[0], waveforms.shape[1], len(indiv_features)))
    for i, feature in enumerate(indiv_features):
        # feature : n_waveforms * n_channels
        features[:, :, i] = feature
    # Reshape so that the features from all channels are combined.
    # Also remove nans.
    return np.nan_to_num(features.reshape((waveforms.shape[0], -1)))


def plot_feature_correlation(features):
    C = np.corrcoef(features, rowvar=False)
    plt.imshow(C)
    plt.show()


ENC = OrdinalEncoder()
ENC.fit([["surface event"], ["earthquake"]])


def get_y(metadata):
    return ENC.transform(metadata.source_type.to_numpy().reshape(-1, 1)).reshape(-1)


def get_X_y(dataset):
    dataset: seisbench.data.WaveformDataset
    print("computing features...")
    X = compute_features(dataset.get_waveforms(dataset.metadata.index))
    print("encoding class labels...")
    y = get_y(dataset.metadata)
    return X, y


def train_RF(X, y):
    clf = RandomForestClassifier(1000)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    print("fitting random forest...")
    clf.fit(X_train, y_train)
    print(f"score on test data = {clf.score(X_test, y_test)}")
    return clf


def plot_feature_importances(clf, X_test, y_test):
    print("calculating permutation importance...")
    imps = permutation_importance(clf, X_test, y_test, n_jobs=-1)
    plt.bar(
        [
            f"{i//13+1} {n}"
            for i, n in enumerate(
                [
                    "min",
                    "max",
                    "mean",
                    "var",
                    "skew",
                    "kurt",
                    "fft_min",
                    "fft_max",
                    "fft_mean",
                    "fft_var",
                    "fft_skew",
                    "fft_kurt",
                    "rand",
                ]
                * 3
            )
        ],
        imps.importances_mean,
        yerr=imps.importances_std,
    )
    plt.show()
