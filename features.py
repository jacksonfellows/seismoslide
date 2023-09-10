import numpy as np
import scipy
import seisbench.data
from matplotlib import pyplot as plt

pnw_exotic = seisbench.data.PNWExotic()
# pnw = seisbench.data.PNW()


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
