import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_X = np.load("train_features.npy")
valid_X = np.load("valid_features.npy")

train_metadata = pd.read_csv("pnw_splits/train/metadata.csv")
valid_metadata = pd.read_csv("pnw_splits/valid/metadata.csv")

train_y = (train_metadata.source_type == "surface event").to_numpy(dtype=int)
valid_y = (valid_metadata.source_type == "surface event").to_numpy(dtype=int)

clf = RandomForestClassifier(1000)
# clf.fit(train_X, train_y)
# clf.score(valid_X, valid_y)
