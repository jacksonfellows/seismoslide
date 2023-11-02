from dataclasses import dataclass, field

import numpy as np

from evolve import EvolutionParameters, load_evolver
from supertree import ClassifierMixin, DecisionTree, build_decision_tree, new_evolver

train = load_evolver("train")
valid = load_evolver("valid")


@dataclass
class RandomForest(ClassifierMixin):
    evolution_params: EvolutionParameters
    n_trees: int
    max_depth: int
    min_gini: float
    trees: list[DecisionTree] = field(default_factory=list)

    def fit(self, W, y, p=1):
        assert len(self.trees) == 0
        for i in range(self.n_trees):
            ii = np.random.permutation(W.shape[0])[: int(W.shape[0] * p)]
            E = new_evolver(W[ii], y[ii])
            print(f"fitting tree {i} w/ {len(ii)} samples")
            self.trees.append(
                build_decision_tree(
                    E, self.evolution_params, self.max_depth, self.min_gini
                )
            )

    def predict_proba(self, W):
        assert len(self.trees) == self.n_trees
        probs = np.zeros((W.shape[0], 2))
        for tree in self.trees:
            probs += tree.predict_proba(W)
        probs /= self.n_trees
        return probs
