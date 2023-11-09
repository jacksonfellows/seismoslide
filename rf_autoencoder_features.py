import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_X = np.load("train_features.npy")
valid_X = np.load("valid_features.npy")

train_metadata = pd.read_csv("pnw_splits/train/metadata.csv")
valid_metadata = pd.read_csv("pnw_splits/valid/metadata.csv")

train_y = (train_metadata.source_type == "surface event").to_numpy(dtype=int)
valid_y = (valid_metadata.source_type == "surface event").to_numpy(dtype=int)

clf = RandomForestClassifier(n_estimators=1000, criterion="entropy")
# clf.fit(train_X, train_y)
# clf.score(valid_X, valid_y)

# Perform hyper-parameter optimization.

rf_param_grid = {
    # "bootstrap": [True, False],
    # "ccp_alpha": [0.0, 0.01, 0.05],
    "criterion": ["gini", "entropy", "log_loss"],
    # "max_depth": None,
    # "max_features": ["sqrt", "log2", 0.1, 0.5, 1.0],
    # "max_leaf_nodes": None,
    # "max_samples": [0.1, 0.5, 1.0],
    # "min_impurity_decrease": 0.0,
    # "min_samples_leaf": 1,
    # "min_samples_split": 2,
    # "min_weight_fraction_leaf": 0.0,
    "n_estimators": [100, 500, 1000, 1500],
}

# (* 2 3 3 5 3 4)


def hyper_parameter_opt():
    gs = GridSearchCV(RandomForestClassifier(), rf_param_grid, n_jobs=-1, cv=4)
    gs.fit(train_X, train_y)
    return gs


# clf2 = GaussianNB() # Doesn't seem to work very well.

# TODO: scale data?
clf2 = SVC(C=10, gamma=0.001)

svc_param_grid = {
    "gamma": [10**n for n in range(-3, 4)],
    "C": [10**n for n in range(-3, 4)],
}


def svc_hyper_opt():
    gs = GridSearchCV(SVC(), svc_param_grid, n_jobs=-1, cv=4)
    gs.fit(train_X, train_y)
    return gs
