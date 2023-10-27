import time

from evolve import *


def run():
    start_t = time.time()
    params = EvolutionParameters(
        pop_size=30,
        n_keep_best=5,
        crossover_rate=0.4,
        mutation_rate=0.2,
        subtree_mutation_rate=0.4,
        max_depth=5,
        n_generations=20,
        terminal_proba=0.2,
        should_contain=None,
    )
    n_features = 20
    print(f"{params=} {n_features=}")
    train = load_evolver("train")
    valid = load_evolver("valid")
    fs = train.meta_evolve_features(params, n_features)
    print(f"{fs=}")
    score_model(train, valid, fs)
    print(f"took {time.time() - start_t} s")


if __name__ == "__main__":
    run()
