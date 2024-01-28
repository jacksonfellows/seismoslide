import numpy as np
import scipy


def taper(waveform, taper_len_s, freq):
    taper_len = taper_len_s * freq
    hann = scipy.signal.windows.hann(2 * taper_len)
    taper = np.ones(waveform.shape[-1])
    taper[0:taper_len] = hann[0:taper_len]
    taper[-taper_len:] = hann[-taper_len:]
    return taper * waveform


def normalize(waveform):
    """Normalize a waveform for testing/prediction."""
    normalized = scipy.signal.detrend(waveform)  # Should I also remove mean?
    stddev = np.std(normalized)
    if stddev != 0:
        normalized /= np.std(normalized)  # std vs max?
    return normalized
