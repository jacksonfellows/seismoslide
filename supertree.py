from dataclasses import dataclass

import graphviz
import numpy as np
from matplotlib import pyplot as plt

import evolve
from operators import all_operators

train = evolve.load_evolver("train")


def gini_impurity(x):
    # Assume x is a vector of 0,1 class labels.
    p1 = np.sum(x) / len(x)
    p0 = 1 - p1
    return 1 - p1 * p1 - p0 * p0


def ginis(x):
    p1s = np.cumsum(x) / np.arange(1, len(x) + 1)
    p0s = 1 - p1s
    return 1 - p1s * p1s - p0s * p0s


def best_split(x):
    """
    Find split that minimizes Gini impurity of both sides.
    Returns a tuple (Gini impurity, split index).
    """
    ginis_left = ginis(x)[:-1]
    ginis_right = ginis(x[::-1])[::-1][1:]
    n = np.arange(1, len(x))
    a = n / len(x) * ginis_left + n[::-1] / len(x) * ginis_right
    i = np.argmin(a)
    return a[i], i + 1


def gini_scorer(E):
    def scorer(features):
        X = E.eval_features(features)
        scores = np.zeros(X.shape[-1])
        for i in range(X.shape[-1]):
            ii = np.argsort(X[:, i])
            y = E.y[ii]
            scores[i] = 0.5 - best_split(y)[0]
        # TODO: Add depth penalty?
        ranks = np.array([E.feature_rank(f) for f in features])
        rank_penalty = 0.5 * (ranks != 0)
        repeated_bandpass_penalty = 0.5 * np.array(
            [evolve.feature_repeated_bandpass(E.simplify_feature(f)) for f in features],
            dtype=float,
        )
        return np.clip(scores - rank_penalty - repeated_bandpass_penalty, 0, None)

    return scorer


def new_evolver(W, y):
    ii = np.random.permutation(np.arange(y.shape[0]))
    return evolve.Evolver(
        W=W[ii, :],
        y=y[ii],
        operators=all_operators,
    )


class ClassifierMixin:
    def predict(self, W):
        return np.argmax(self.predict_proba(W), axis=-1)

    def score(self, W, y):
        pred_y = self.predict(W)
        return np.sum(pred_y == y) / len(y)


@dataclass
class DecisionTree(ClassifierMixin):
    feature: tuple
    threshold: float
    gini: float
    samples: int
    left: any
    right: any

    def plot(self):
        G = graphviz.Graph()
        self.plot_rec(G)
        return G

    def plot_rec(self, G: graphviz.Graph):
        i = str(id(self))
        G.node(
            i,
            label=f"{self.feature}, threshold={self.threshold:0.3f}, gini={self.gini:0.3f}, samples={self.samples}",
        )
        G.edges(((i, self.left.plot_rec(G)), (i, self.right.plot_rec(G))))
        return i

    def predict_proba(self, W):
        # W : n_traces,n_components,n_samples
        # Don't use new_evolver() here as it permutes the samples!
        E = evolve.Evolver(W=W, y=None, operators=all_operators)
        X = E.eval_feature(self.feature)
        left = X < self.threshold
        right = ~left
        probs = np.full((W.shape[0], 2), np.nan)
        probs[left] = self.left.predict_proba(W[left, :, :])
        probs[right] = self.right.predict_proba(W[right, :, :])
        return probs


@dataclass
class TerminalNode(ClassifierMixin):
    n_ones: int
    samples: int
    gini: float

    def plot_rec(self, G: graphviz.Graph):
        i = str(id(self))
        G.node(
            i,
            label=f"p1={self.n_ones/self.samples}, gini={self.gini}, samples={self.samples}",
        )
        return i

    def predict_proba(self, W):
        p_one = self.n_ones / self.samples
        return np.column_stack(
            (
                np.full(W.shape[0], 1 - p_one),  # prob 0
                np.full(W.shape[0], p_one),  # prob 1
            )
        )
        return np.full(W.shape[0], self.pred_class)


# E10 = new_evolver(train.W[0:10], train.y[0:10])

params = evolve.EvolutionParameters(
    pop_size=20,
    n_keep_best=5,
    crossover_rate=0.4,
    mutation_rate=0.2,
    subtree_mutation_rate=0.4,
    max_depth=5,
    n_generations=10,
    terminal_proba=0.2,
    should_contain=None,
)


def build_decision_tree(
    E, evolution_params: evolve.EvolutionParameters, max_depth, min_gini=0
):
    if len(E.y) == 0:
        return None
    whole_gini = gini_impurity(E.y)
    if max_depth == 0 or whole_gini <= min_gini:
        n_ones = len(E.y[E.y == 1])
        return TerminalNode(samples=len(E.y), gini=whole_gini, n_ones=n_ones)
    best_feature = E.simplify_feature(
        E.evolve_features(evolution_params, fitness_f=gini_scorer(E))[0]
    )
    print(f"{best_feature=}")
    X = E.eval_feature(best_feature)
    ii = np.argsort(X)
    gini, split_i = best_split(E.y[ii])
    threshold = X[ii][split_i]
    samples = len(E.y)
    print(f"{gini=}, {split_i=}, {threshold=}, {samples=}")
    return DecisionTree(
        feature=best_feature,
        threshold=threshold,
        gini=gini,
        samples=samples,
        left=build_decision_tree(
            new_evolver(E.W[ii][:split_i], E.y[ii][:split_i]),
            evolution_params,
            max_depth - 1,
            min_gini=min_gini,
        ),
        right=build_decision_tree(
            new_evolver(E.W[ii][split_i:], E.y[ii][split_i:]),
            evolution_params,
            max_depth - 1,
            min_gini=min_gini,
        ),
    )
