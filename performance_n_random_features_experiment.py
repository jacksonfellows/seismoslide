import os

from evolve import default_evolver

n_samples = 10
n_random_features = list(range(5, 101, 5))

max_depth = 5
terminal_proba = 0.3

path = "n_random_features_vs_model_score_2.csv"

if os.path.exists(path):
    print(f"{path} already exists!")
else:
    with open(path, "w") as results_file:
        results_file.write("n_random_features,model_score,features\n")
        for n_random in n_random_features:
            for _ in range(n_samples):
                features = [
                    default_evolver.random_feature(
                        max_depth=max_depth, terminal_proba=terminal_proba
                    )
                    for _ in range(n_random)
                ]
                print(f"{features=}")
                score = default_evolver.score_model(features, n_jobs=-1)
                results_file.write(f'{n_random},{score:0.3f},"{features}"\n')
                results_file.flush()
