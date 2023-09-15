import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
import seisbench.data
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

dataset = seisbench.data.WaveformDataset(
    Path.home() / ".seisbench/datasets/seismoslide_1"
)
W = dataset.get_waveforms(mask=np.ones(len(dataset), dtype=bool))
y = (dataset.metadata.source_type == "surface event").to_numpy(dtype=int)


RANDOM_STATE = 2001

unary_funcs = {
    "min": lambda x: np.min(x, axis=-1),
    "max": lambda x: np.max(x, axis=-1),
    "mean": lambda x: np.mean(x, axis=-1),
    "median": lambda x: np.median(x, axis=-1),
    "skew": lambda x: scipy.stats.skew(x, axis=-1),
    "kurtosis": lambda x: scipy.stats.kurtosis(x, axis=-1),
    "re_fft": lambda x: np.real(scipy.fft.fft(x, axis=-1)),
    "im_fft": lambda x: np.imag(scipy.fft.fft(x, axis=-1)),
}

unary_dranks = {
    "min": -1,
    "max": -1,
    "mean": -1,
    "median": -1,
    "skew": -1,
    "kurtosis": -1,
    "re_fft": 0,
    "im_fft": 0,
}

@lru_cache(maxsize=256)         # TODO: This is not the optimal caching strategy.
def eval_feature_rec(feature):
    match feature:
        case (f, x):
            x_ = eval_feature_rec(x)
            if len(x_.shape) + unary_dranks[f] < 2:
                return x_
            return np.nan_to_num(unary_funcs[f](x_))
        case "x": return W

def eval_feature(feature):
    r = eval_feature_rec(feature)
    if len(r.shape) > 2:
        return r[:, :, 0]
    return r

def eval_features(features):
    X = np.zeros((*W.shape[:-1], len(features)))
    for i,feature in enumerate(features):
        X[:, :, i] = eval_feature(feature)
    return X.reshape((W.shape[0], -1))

def score_features(features):
    scores = mutual_info_classif(eval_features(features), y, random_state=RANDOM_STATE)
    return scores.reshape((3, -1)).mean(axis=0)

def random_feature(max_depth):
    if max_depth == 0 or random.random() < 0.3:
        return "x"
    return (random.choice(list(unary_funcs.keys())), random_feature(max_depth - 1))

def evolve_features():
    pop_size = 30
    keep_best = 5
    max_depth = 5
    population = [random_feature(max_depth) for _ in range(pop_size)]
    n_generations = 10
    for g in range(n_generations):
        print(f"generation {g}")
        print("scoring features...")
        fitnesses = score_features(population)
        print(f"max fitness = {fitnesses.max()}, mean fitness = {fitnesses.mean()}")
        new_population = []
        for i in np.argsort(fitnesses)[-keep_best:]:
            new_population.append(population[i])
        for _ in range(pop_size - keep_best):
            new_population.append(random_feature(max_depth))
        assert len(new_population) == pop_size
        population = new_population
    return population

def score_model(features):
    print("computing X...")
    X = eval_features(features)
    clf = RandomForestClassifier(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("fitting model...")
    clf.fit(X_train, y_train)
    print("scoring model...")
    score = clf.score(X_test, y_test)
    print(f"{score=}")
