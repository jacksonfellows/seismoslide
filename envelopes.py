from pathlib import Path

import numpy as np
import scipy
import seisbench.data


# Taken from GP operators.
def envelope(x):
    return np.abs(scipy.signal.hilbert(x, axis=-1))


def write_envelopes(split):
    split_dir = f"./pnw_splits/{split}"
    dataset = seisbench.data.WaveformDataset(split_dir)
    spectrogram_path = Path(f"./pnw_splits_Z_envelopes/{split}")
    for i in range(len(dataset)):
        waveforms = dataset.get_waveforms(i)
        Z = waveforms[0].astype("float32")
        E = envelope(Z)
        np.save(spectrogram_path / str(i), E)


def write_all():
    for split in ["train", "valid", "test"]:
        write_envelopes(split)
