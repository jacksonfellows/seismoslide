import numpy as np
import scipy


def normalize(waveform):
    """Normalize a waveform for testing/prediction."""
    normalized = scipy.signal.detrend(waveform, axis=-1, type="constant")
    return normalized / np.std(normalized, axis=-1)[:, None]
