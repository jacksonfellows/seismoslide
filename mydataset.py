from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# A dataset is a directory with ./metadata.csv and each sample as ./data/XXX.npy.


class MyDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = pd.read_csv(self.root / "metadata.csv", index_col=0)
        self.data_path = self.root / "data"

    def __getitem__(self, i):
        # Needed for PyTorch Dataset.
        return np.load(self.data_path / f"{i}.npy")

    def __len__(self, i):
        # Needed for PyTorch Dataset.
        return len(self.metadata)


class MyDatasetWriter:
    def __init__(self, root, metadata_columns):
        self.root = Path(root)
        if self.root.exists():
            raise ValueError(f"directory {root} already exists")
        self.root.mkdir()
        self.data_path = root / "data"
        self.data_path.mkdir()
        self.metadata = pd.DataFrame(columns=metadata_columns)

    def write_sample(self, sample, sample_metadata):
        i = len(self.metadata)
        self.metadata.loc[i] = [sample_metadata[col] for col in self.metadata.columns]
        np.save(self.data_path / str(i), sample)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.metadata.to_csv(self.root / "metadata.csv")
