import random
import time
from collections import namedtuple
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
import scipy
import seisbench.data
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

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
        "terminal_proba",
    ),
)

OperatorDef = namedtuple("OperatorDef", ("arity", "delta_rank", "apply", "cell_rank"))


class Evolver:
    def __init__(self, W, y, operators):
        self.W = W
        self.y = y
        self.operators = operators
        # Hard-code for now.
        self.terminals = {
            "Z": W[:, 0],
            "N": W[:, 1],
            "E": W[:, 2],
        }

    def simplify_feature_rec(self, feature):
        match feature:
            case (f, *x):
                simplifieds, ranks = zip(*map(self.simplify_feature_rec, x))
                max_rank = max(ranks)
                if max_rank == self.operators[f].cell_rank:
                    # Would combine across traces - Not allowed.
                    return simplifieds[0], max_rank
                if max_rank + self.operators[f].delta_rank < 1:
                    # No-op.
                    return simplifieds[0], max_rank
                return (f, *simplifieds), max_rank + self.operators[f].delta_rank
            case x:
                return x, len(self.terminals[x].shape)

    def simplify_feature(self, feature):
        return self.simplify_feature_rec(feature)[0]

    def feature_rank(self, feature):
        return self.simplify_feature_rec(feature)[1] - 1

    @lru_cache(maxsize=64)  # TODO: This is not the optimal caching strategy.
    def eval_feature_rec(self, feature):
        # print(f"eval_feature_rec({feature})")
        match feature:
            case (f, *x):
                x_ = list(map(self.eval_feature_rec, x))
                max_rank = max(len(x__.shape) for x__ in x_)
                if max_rank == self.operators[f].cell_rank:
                    # Would combine across traces - Not allowed.
                    return x_[0]
                if max_rank + self.operators[f].delta_rank < 1:
                    # No-op.
                    return x_[0]
                return np.nan_to_num(self.operators[f].apply(*x_))
            case x:
                return self.terminals[x]

    def eval_feature(self, feature, simplify_first=True):
        if simplify_first:
            feature = self.simplify_feature(feature)
        x = self.eval_feature_rec(feature)
        while len(x.shape) > 1:  # TODO: stupid
            x = x[:, 0]
        return x

    def eval_features(self, features):
        print(f"evaluating {len(features)} features")
        X = np.zeros((self.W.shape[0], len(features)))
        for i, feature in enumerate(features):
            # print(f"evaluating feature {i}")
            X[:, i] = self.eval_feature(feature)
        return X

    def score_features(self, features):
        X = self.eval_features(features)
        print("calculating mutual info...")
        mi = mutual_info_classif(X, self.y, random_state=RANDOM_STATE)
        n_subtrees = np.array([self.count_subtrees(f) for f in features])
        ranks = np.array([self.feature_rank(f) for f in features])
        depth_penalty = 0.3 / (1 + np.exp(10 - 0.6 * n_subtrees))
        rank_penalty = 0.3 * (ranks != 0)
        return np.clip(mi - depth_penalty - rank_penalty, 0, None)

    def score_features_similarity(self, features, existing_features):
        A = self.eval_features(features)
        B = self.eval_features(existing_features)
        C = np.abs(np.nan_to_num(np.corrcoef(A, B, rowvar=False)))
        # Does max make sense?
        return C[: len(features), -len(existing_features) :].max(axis=-1)

    def score_features_with_existing(self, features, existing_features):
        score = self.score_features(features)  # mutual importance
        if len(existing_features) == 0:
            return score
        simil = self.score_features_similarity(
            features, existing_features
        )  # absolute correlation [0,1]
        return np.clip(score - 0.3 * simil, 0, None)

    def random_feature(self, max_depth, terminal_proba):
        if max_depth == 0 or random.random() < terminal_proba:
            return random.choice(list(self.terminals.keys()))
        head = random.choice(list(self.operators.keys()))
        return (
            head,
            *[
                self.random_feature(max_depth - 1, terminal_proba)
                for _ in range(self.operators[head].arity)
            ],
        )

    def point_mutate_feature(self, feature, p=0.3):
        match feature:
            case (f, *x):
                if random.random() < p:
                    return (
                        random.choice(
                            [
                                op
                                for op, d in self.operators.items()
                                if d.arity == self.operators[f].arity
                            ]
                        ),
                        *map(self.point_mutate_feature, x),
                    )
                return (f, *map(self.point_mutate_feature, x))
            case x:
                if random.random() < p:
                    return random.choice(list(self.terminals.keys()))
                else:
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
            self.random_feature(parameters.max_depth, parameters.terminal_proba)
            for _ in range(parameters.pop_size)
        ]
        for g in range(parameters.n_generations):
            print(f"generation {g}")
            print("scoring features...")
            fitnesses = fitness_f(population)
            print(
                f"max fitness={fitnesses.max():0.3f}, mean fitness={fitnesses.mean():0.3f}, best feature={self.simplify_feature(population[np.argmax(fitnesses)])}"
            )
            if fitnesses.sum() == 0:
                fitnesses += (
                    0.001  # random.choice doesn't like when the sum of weights is <= 0.
                )
            new_population = []
            for i in reversed(np.argsort(fitnesses)[-parameters.n_keep_best :]):
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
                        self.point_mutate_feature(
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

    def score_model(self, features, n_jobs=None):
        print("computing X...")
        X = self.eval_features(features)
        clf = RandomForestClassifier(1000, random_state=RANDOM_STATE)
        n_splits = 8
        test_size = 0.25
        print(f"cross validating model (shuffle splits, {n_splits=}, {test_size=})")
        scores = cross_val_score(
            clf,
            X,
            self.y,
            cv=ShuffleSplit(n_splits=n_splits, test_size=test_size),
            n_jobs=n_jobs,
        )
        mean_score = scores.mean()
        print(f"mean score={mean_score}")
        return mean_score

    def calculate_permutation_importances(self, features, n_jobs=None):
        print("computing X...")
        X = self.eval_features(features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, random_state=RANDOM_STATE
        )
        print("fitting model...")
        clf = RandomForestClassifier(1000, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        print("scoring model...")
        score = clf.score(X_test, y_test)
        print(f"{score=}")
        n_repeats = 8
        print(f"calculating permutation importances ({n_repeats=})...")
        imps = permutation_importance(
            clf, X_test, y_test, n_repeats=n_repeats, n_jobs=n_jobs
        )
        print(f"{imps=}")
        return imps


# Operators.
def make_same_rank(f):
    @wraps(f)
    def wrapper(*args):
        max_rank = max(len(x.shape) for x in args)
        new_args = []
        for arg in args:
            while len(arg.shape) < max_rank:
                arg = np.expand_dims(arg, -1)
            new_args.append(arg)
        return f(*new_args)

    return wrapper


@make_same_rank
def safe_add(x, y):
    if x.shape[-1] < y.shape[-1]:
        y = y[..., : x.shape[-1]]
    elif y.shape[-1] < x.shape[-1]:
        x = x[..., : y.shape[-1]]
    return x + y


@make_same_rank
def safe_correlate(x, y):
    res = np.empty_like(x)
    for i in np.ndindex(res.shape[:-1]):
        res[i] = scipy.signal.correlate(x[i], y[i], mode="same")
    return res


def first_half(x):
    if x.shape[-1] == 1:
        return x
    return x[..., : x.shape[-1] // 2]


def second_half(x):
    return x[..., x.shape[-1] // 2 :]


def envelope(x):
    return np.abs(scipy.signal.hilbert(x, axis=-1))


def periodogram(x):
    return scipy.signal.periodogram(x, axis=-1)[1]


dataset = seisbench.data.WaveformDataset(
    Path.home() / ".seisbench/datasets/seismoslide_1"
)
default_evolver = Evolver(
    W=dataset.get_waveforms(mask=np.ones(len(dataset), dtype=bool)),
    y=(dataset.metadata.source_type == "surface event").to_numpy(dtype=int),
    operators={
        "min": OperatorDef(
            arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.min(x, axis=-1)
        ),
        "max": OperatorDef(
            arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.max(x, axis=-1)
        ),
        "argmax": OperatorDef(
            arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.argmax(x, axis=-1)
        ),
        "skew": OperatorDef(
            arity=1,
            delta_rank=-1,
            cell_rank=1,
            apply=lambda x: scipy.stats.skew(x, axis=-1),
        ),
        "kurtosis": OperatorDef(
            arity=1,
            delta_rank=-1,
            cell_rank=1,
            apply=lambda x: scipy.stats.kurtosis(x, axis=-1),
        ),
        "re_fft": OperatorDef(
            arity=1,
            delta_rank=0,
            cell_rank=1,
            apply=lambda x: np.real(scipy.fft.fft(x, axis=-1)),
        ),
        # "im_fft": OperatorDef(
        #     arity=1, delta_rank=0, cell_rank=1, apply=lambda x: np.imag(scipy.fft.fft(x, axis=-1))
        # ),
        "+": OperatorDef(arity=2, delta_rank=0, cell_rank=0, apply=safe_add),
        "corr": OperatorDef(arity=2, delta_rank=0, cell_rank=1, apply=safe_correlate),
        "first_half": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=first_half),
        "second_half": OperatorDef(
            arity=1, delta_rank=0, cell_rank=1, apply=second_half
        ),
        "sum": OperatorDef(
            arity=1, delta_rank=-1, cell_rank=1, apply=lambda x: np.sum(x, axis=-1)
        ),
        "envelope": OperatorDef(arity=1, delta_rank=0, cell_rank=1, apply=envelope),
        "periodogram": OperatorDef(
            arity=1, delta_rank=0, cell_rank=1, apply=periodogram
        ),
    },
)


def benchmark_default_single():
    start_t = time.time()
    params = EvolutionParameters(
        pop_size=40,
        n_keep_best=10,
        crossover_rate=0.4,
        mutation_rate=0.4,
        max_depth=7,
        n_generations=30,
        terminal_proba=0.5,
    )
    print(f"{params=}")
    fs = default_evolver.evolve_features(
        params, fitness_f=default_evolver.score_features
    )
    best_feature = default_evolver.simplify_feature(fs[0])
    print(f"best feature: {best_feature}")
    print(f"took {time.time() - start_t} s")


def benchmark_default():
    start_t = time.time()
    params = EvolutionParameters(
        pop_size=40,
        n_keep_best=5,
        crossover_rate=0.5,
        mutation_rate=0.4,
        max_depth=10,
        n_generations=20,
        terminal_proba=0.3,
    )
    n_features = 30
    print(f"{params=} {n_features=}")
    fs = default_evolver.meta_evolve_features(params, n_features)
    print(f"{fs=}")
    default_evolver.score_model(fs)
    print(f"took {time.time() - start_t} s")
