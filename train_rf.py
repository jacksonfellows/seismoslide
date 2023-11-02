import pickle
import sys
from pathlib import Path

from random_forest import *

rf = RandomForest(
    n_trees=10,
    max_depth=5,
    min_gini=0.01,
    evolution_params=EvolutionParameters(
        pop_size=20,
        n_keep_best=4,
        crossover_rate=0.4,
        mutation_rate=0.2,
        subtree_mutation_rate=0.2,
        max_depth=5,
        n_generations=30,
        terminal_proba=0.2,
        should_contain=None,
    ),
)


if __name__ == "__main__":
    n = sys.argv[1]
    path = Path(f"rf{n}.pickle")
    print(f"{path=}")
    if not path.exists():
        rf.fit(train.W, train.y, p=0.1)
        score_train = rf.score(train.W, train.y)
        print(f"{score_train=}")
        score_valid = rf.score(valid.W, valid.y)
        print(f"{score_valid=}")
        with open(path, "wb") as f:
            pickle.dump(rf, f)
    else:
        print(f"{path} already exists!")
