import itertools
import random
import time
from collections import namedtuple
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import seisbench.data
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from operators import all_operators

RANDOM_STATE = 2001


EvolutionParameters = namedtuple(
    "EvolutionParameters",
    (
        "pop_size",
        "n_keep_best",
        "crossover_rate",
        "mutation_rate",
        "subtree_mutation_rate",
        "max_depth",
        "n_generations",
        "terminal_proba",
        "should_contain",
    ),
)


def feature_contains(feature, operator):
    match feature:
        case (f, *x):
            if f == operator or (
                type(operator) in {set, tuple, list} and f in operator
            ):
                return True
            return any(feature_contains(x_, operator) for x_ in x)
        case x:  # Don't count terminals.
            return False


def feature_repeated_bandpass(feature):
    # Kind of silly.
    match feature:
        case (f, *x):
            if f.startswith("bp_") and x[0][0].startswith("bp_"):
                return True
            return any(feature_repeated_bandpass(x_) for x_ in x)
        case _:
            return False


def count_subtrees(feature):
    match feature:
        case (_, *x):
            return 1 + sum(map(count_subtrees, x))
        case x:
            return 1


# def feature_max_depth(feature):
#     def rec(tree, depth):
#         match tree:
#             case (_, *x):
#                 return max(rec(x_, depth + 1) for x_ in x)
#             case _:
#                 return depth

#     return rec(feature, 0)


class Evolver:
    def __init__(self, W, y, operators, operator_weights=None):
        self.W = W
        self.y = y
        self.operators = operators
        self.operator_weights = (
            [1 for _ in operators] if operator_weights is None else operator_weights
        )
        # Hard-code for now.
        self.terminals = {
            "Z": W[:, 0],
            "N": W[:, 1],
            "E": W[:, 2],
        }
        for component, i in itertools.product("ZNE", range(1, 4)):
            T = self.terminals[component]
            s = T.shape[-1] // 3
            self.terminals[component + str(i)] = T[:, (i - 1) * s : i * s]

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

    @lru_cache(maxsize=128)
    def eval_feature_rec_cached(self, feature):
        f, *x = feature
        x_ = list(map(self.eval_feature_rec, x))
        return np.nan_to_num(self.operators[f].apply(*x_))

    def eval_feature_rec(self, feature):
        # Assume feature is simplified!
        match feature:
            case (f, *x):
                if f in {
                    # "skew",
                    # "kurtosis",
                    "abs_fft",
                    "envelope",
                    "corr",
                    "bp_1_3",
                    "bp_3_10",
                    "bp_10_20",
                    "bp_20_45",
                    "spectrogram",
                }:
                    return self.eval_feature_rec_cached(feature)
                x_ = list(map(self.eval_feature_rec, x))
                return np.nan_to_num(self.operators[f].apply(*x_))
            case x:
                return self.terminals[x]

    def eval_feature(self, feature):
        x = self.eval_feature_rec(self.simplify_feature(feature))
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

    def score_features(self, features, should_contain=None):
        X = self.eval_features(features)
        # print("calculating mutual info...")
        # mi = mutual_info_classif(X, self.y, random_state=RANDOM_STATE)
        window_len = 1000
        mi = [
            np.mean(
                np.abs(
                    np.sum(
                        np.lib.stride_tricks.sliding_window_view(
                            self.y[np.argsort(X[:, i])], window_len
                        ),
                        axis=-1,
                    )
                    / window_len
                    - 0.5
                )
            )
            for i in range(len(features))
        ]
        n_subtrees = np.array([count_subtrees(f) for f in features])
        ranks = np.array([self.feature_rank(f) for f in features])
        depth_penalty = 0.3 / (1 + np.exp(10 - 0.6 * n_subtrees))
        rank_penalty = 0.3 * (ranks != 0)
        repeated_bandpass_penalty = 0.5 * np.array(
            [feature_repeated_bandpass(self.simplify_feature(f)) for f in features],
            dtype=float,
        )
        scores = mi - depth_penalty - rank_penalty - repeated_bandpass_penalty
        if should_contain is not None:
            contains_penalty = 0.5 * np.array(
                [
                    not feature_contains(self.simplify_feature(f), should_contain)
                    for f in features
                ],
                dtype=float,
            )
            scores -= contains_penalty
        return np.clip(scores, 0, None), X

    def score_features_similarity(self, A, existing_features):
        B = self.eval_features(existing_features)
        C = np.abs(np.nan_to_num(np.corrcoef(A, B, rowvar=False)))
        # Does max make sense?
        return C[: A.shape[-1], -len(existing_features) :].max(axis=-1)

    def score_features_with_existing(
        self, features, existing_features, should_contain=None
    ):
        score, X = self.score_features(features, should_contain)  # mutual importance
        if len(existing_features) == 0:
            return score
        simil = self.score_features_similarity(
            X, existing_features
        )  # absolute correlation [0,1]
        return np.clip(score - 0.3 * simil, 0, None)

    def random_operator(self, operators=None):
        if operators is None:
            operators = list(self.operators.keys())
        return random.choices(
            operators,
            weights=[
                w
                for o, w in zip(self.operators.keys(), self.operator_weights)
                if o in operators
            ],
            k=1,
        )[0]

    def random_feature(self, max_depth, terminal_proba):
        if max_depth == 0 or random.random() < terminal_proba:
            return random.choice(list(self.terminals.keys()))
        head = self.random_operator()
        return (
            head,
            *[
                self.random_feature(max_depth - 1, terminal_proba)
                for _ in range(self.operators[head].arity)
            ],
        )

    def subtree_mutate_feature(self, feature, max_depth, terminal_proba):
        i = random.randrange(count_subtrees(feature))
        random_subtree = self.random_feature(max_depth, terminal_proba)
        return self.swap_subtree(feature, i, random_subtree)

    def point_mutate_feature(self, feature, p=0.3):
        match feature:
            case (f, *x):
                if random.random() < p:
                    return (
                        self.random_operator(
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
        i1 = random.randrange(count_subtrees(feature1))
        i2 = random.randrange(count_subtrees(feature2))
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
                r = random.random()
                if r < parameters.crossover_rate:
                    new_population.extend(
                        self.crossover(
                            *random.choices(population, weights=fitnesses, k=2)
                        )
                    )
                elif r < parameters.crossover_rate + parameters.mutation_rate:
                    new_population.append(
                        self.point_mutate_feature(
                            random.choices(population, weights=fitnesses, k=1)[0]
                        )
                    )
                elif (
                    r
                    < parameters.crossover_rate
                    + parameters.mutation_rate
                    + parameters.subtree_mutation_rate
                ):
                    new_population.append(
                        self.subtree_mutate_feature(
                            random.choices(population, weights=fitnesses, k=1)[0],
                            parameters.max_depth,
                            parameters.terminal_proba,
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
                    x, existing_features, parameters.should_contain
                ),
            )[0]
            new_feature = self.simplify_feature(new_feature)
            print(f"adding feature {new_feature}")
            existing_features.append(new_feature)
        return existing_features


def score_model(
    evolver_train,
    evolver_test,
    features,
    n_trees=1000,
    n_jobs=None,
    calculate_permutation_importances=False,
    n_repeats=8,
):
    print("computing train X...")
    X_train = evolver_train.eval_features(features)
    y_train = evolver_train.y
    print("computing test X...")
    X_test = evolver_test.eval_features(features)
    y_test = evolver_test.y
    print(f"fitting random forest ({n_trees=})...")
    clf = RandomForestClassifier(n_trees, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print("scoring random forest...")
    score = clf.score(X_test, y_test)
    print(f"score={score:0.4f}")
    if calculate_permutation_importances:
        print(f"calculating permutation importances ({n_repeats=})...")
        imps = permutation_importance(
            clf, X_test, y_test, n_repeats=n_repeats, n_jobs=n_jobs
        )
        return imps
    return score


@cache
def load_evolver(split):
    dataset = seisbench.data.WaveformDataset(f"./pnw_splits/{split}")
    W = dataset.get_waveforms(mask=np.ones(len(dataset), dtype=bool))
    y = (dataset.metadata.source_type == "surface event").to_numpy(dtype=int)
    ii = np.random.permutation(np.arange(y.shape[0]))
    return Evolver(
        W=W[ii, :],
        y=y[ii],
        operators=all_operators,
    )
