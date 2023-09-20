import random
import time
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
import seisbench.data
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 2001


EvolutionParameters = namedtuple(
    "EvolutionParameters",
    (
        "pop_size",
        "n_keep_best",
        "crossover_rate",
        "mutation_rate",
        "max_depth",
        "n_generations",
    ),
)


class Evolver:
    def __init__(self, W, y, unary_funcs, unary_dranks):
        self.W = W
        self.y = y
        self.unary_funcs = unary_funcs
        self.unary_dranks = unary_dranks

    def simplify_feature(self, feature):
        def rec(tree):
            match tree:
                case (f, x):
                    simplified, rank = rec(x)
                    if rank + self.unary_dranks[f] < 2:
                        return simplified, rank
                    return (f, simplified), rank + self.unary_dranks[f]
                case "x":
                    return "x", 3

        return rec(feature)[0]

    @lru_cache(maxsize=256)  # TODO: This is not the optimal caching strategy.
    def eval_feature_rec(self, feature):
        match feature:
            case (f, x):
                x_ = self.eval_feature_rec(x)
                if len(x_.shape) + self.unary_dranks[f] < 2:
                    return x_
                return np.nan_to_num(self.unary_funcs[f](x_))
            case "x":
                return self.W

    def eval_feature(self, feature, simplify_first=True):
        if simplify_first:
            feature = self.simplify_feature(feature)
        r = self.eval_feature_rec(feature)
        if len(r.shape) > 2:
            return r[:, :, 0]
        return r

    def eval_features(self, features):
        X = np.zeros((*self.W.shape[:-1], len(features)))
        for i, feature in enumerate(features):
            X[:, :, i] = self.eval_feature(feature)
        return X.reshape((self.W.shape[0], -1))

    def score_features(self, features):
        scores = mutual_info_classif(
            self.eval_features(features), self.y, random_state=RANDOM_STATE
        )
        return scores.reshape((3, -1)).mean(axis=0)

    def score_features_with_existing(self, features, existing_features):
        features_X = self.eval_features(features)
        target_scores = mutual_info_classif(
            features_X, self.y, random_state=RANDOM_STATE
        )
        if len(existing_features) == 0:
            return target_scores.reshape((3, -1)).mean(axis=0)
        existing_Xs = self.eval_features(existing_features)
        existing_scores = np.zeros(target_scores.shape)
        for i in range(existing_Xs.shape[-1]):
            existing_scores += mutual_info_regression(
                features_X, existing_Xs[:, i], random_state=RANDOM_STATE
            )
        target_scores = target_scores.reshape((3, -1)).mean(axis=0)
        existing_scores = existing_scores.reshape((3, -1)).mean(axis=0)
        return target_scores * (1 - existing_scores / existing_scores.max())

    def random_feature(self, max_depth):
        if max_depth == 0 or random.random() < 0.3:
            return "x"
        return (
            random.choice(list(self.unary_funcs.keys())),
            self.random_feature(max_depth - 1),
        )

    def mutate_feature(self, feature, p=0.3):
        match feature:
            case (f, x):
                if random.random() < p:
                    return (
                        random.choice(list(self.unary_funcs.keys())),
                        self.mutate_feature(x),
                    )
                return (f, self.mutate_feature(x))
            case x:
                return x

    def count_subtrees(self, feature):
        match feature:
            case (_, *x):
                return 1 + sum(map(self.count_subtrees, x))
            case x:
                return 1

    def get_subtree(self, feature, index):
        # Kind of ugly.
        i = 0

        def rec(tree):
            nonlocal i
            if i == index:
                return tree
            match tree:
                case (_, *x):
                    i += 1
                    for x_ in x:
                        r = rec(x_)
                        if r is not None:
                            return r
                case x:
                    i += 1
                    return None

        return rec(feature)

    def swap_subtree(self, feature, index, subtree):
        # Kind of ugly.
        i = 0

        def rec(tree):
            nonlocal i
            if i == index:
                i += 1
                return subtree
            match tree:
                case (f, *x):
                    i += 1
                    return (f, *[rec(x_) for x_ in x])
                case x:
                    i += 1
                    return x

        return rec(feature)

    def crossover(self, feature1, feature2):
        i1 = random.randrange(self.count_subtrees(feature1))
        i2 = random.randrange(self.count_subtrees(feature2))
        subtree1 = self.get_subtree(feature1, i1)
        subtree2 = self.get_subtree(feature2, i2)
        return self.swap_subtree(feature1, i1, subtree2), self.swap_subtree(
            feature2, i2, subtree1
        )

    def evolve_features(self, parameters: EvolutionParameters, fitness_f):
        population = [
            self.random_feature(parameters.max_depth)
            for _ in range(parameters.pop_size)
        ]
        for g in range(parameters.n_generations):
            print(f"generation {g}")
            print("scoring features...")
            fitnesses = fitness_f(population)
            print(f"max fitness = {fitnesses.max()}, mean fitness = {fitnesses.mean()}")
            new_population = []
            for i in np.argsort(fitnesses)[-parameters.n_keep_best :]:
                new_population.append(population[i])
            while len(new_population) < parameters.pop_size:
                if random.random() < parameters.crossover_rate:
                    new_population.extend(
                        self.crossover(
                            *random.choices(population, weights=fitnesses, k=2)
                        )
                    )
                elif random.random() < parameters.mutation_rate:
                    new_population.append(
                        self.mutate_feature(
                            random.choices(population, weights=fitnesses, k=1)[0]
                        )
                    )
                else:
                    new_population.append(
                        random.choices(population, weights=fitnesses, k=1)[0]
                    )
            population = new_population
        return population

    def meta_evolve_features(self, parameters: EvolutionParameters, n_features):
        existing_features = []
        for _ in range(n_features):
            new_feature = self.evolve_features(
                parameters,
                fitness_f=lambda x: self.score_features_with_existing(
                    x, existing_features
                ),
            )[0]
            new_feature = self.simplify_feature(new_feature)
            print(f"adding feature {new_feature}")
            existing_features.append(new_feature)
        return existing_features

    def score_model(self, features):
        print("computing X...")
        X = self.eval_features(features)
        clf = RandomForestClassifier(1000, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, random_state=RANDOM_STATE
        )
        print("fitting model...")
        clf.fit(X_train, y_train)
        print("scoring model...")
        score = clf.score(X_test, y_test)
        print(f"{score=}")


dataset = seisbench.data.WaveformDataset(
    Path.home() / ".seisbench/datasets/seismoslide_1"
)
default_evolver = Evolver(
    W=dataset.get_waveforms(mask=np.ones(len(dataset), dtype=bool)),
    y=(dataset.metadata.source_type == "surface event").to_numpy(dtype=int),
    unary_funcs={
        "min": lambda x: np.min(x, axis=-1),
        "max": lambda x: np.max(x, axis=-1),
        "mean": lambda x: np.mean(x, axis=-1),
        "median": lambda x: np.median(x, axis=-1),
        "skew": lambda x: scipy.stats.skew(x, axis=-1),
        "kurtosis": lambda x: scipy.stats.kurtosis(x, axis=-1),
        "re_fft": lambda x: np.real(scipy.fft.fft(x, axis=-1)),
        "im_fft": lambda x: np.imag(scipy.fft.fft(x, axis=-1)),
    },
    unary_dranks={
        "min": -1,
        "max": -1,
        "mean": -1,
        "median": -1,
        "skew": -1,
        "kurtosis": -1,
        "re_fft": 0,
        "im_fft": 0,
    },
)


def benchmark_default_single():
    start_t = time.time()
    params = EvolutionParameters(
        pop_size=15,
        n_keep_best=3,
        crossover_rate=0.7,
        mutation_rate=0.1,
        max_depth=5,
        n_generations=10,
    )
    print(f"{params=}")
    fs = default_evolver.evolve_features(
        params, fitness_f=default_evolver.score_features
    )
    print(f"best feature: {default_evolver.simplify_feature(fs[0])}")
    print(f"took {time.time() - start_t} s")


def benchmark_default():
    start_t = time.time()
    params = EvolutionParameters(
        pop_size=7,
        n_keep_best=2,
        crossover_rate=0.7,
        mutation_rate=0.1,
        max_depth=5,
        n_generations=5,
    )
    n_features = 5
    print(f"{params=} {n_features=}")
    fs = default_evolver.meta_evolve_features(params, n_features)
    default_evolver.score_model(fs)
    print(f"took {time.time() - start_t} s")
