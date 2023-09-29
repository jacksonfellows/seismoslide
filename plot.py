import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evolve import default_evolver

df = pd.read_csv("n_random_features_vs_model_score_cleanup.csv")


def p():
    plt.plot(df.n_random_features, df.model_score, "x")
    plt.xlabel("# of Random Features (max_depth=5, terminal_proba=0.3)")
    plt.ylabel("Model Performance (n_splits=8, test_size=0.25)")
    plt.show()


def plot_feature_importances(fs):
    labels = [repr(default_evolver.simplify_feature(f)) for f in fs]
    imps = default_evolver.calculate_permutation_importances(fs, n_jobs=-1)
    x = range(len(labels))
    C = plt.bar(x, imps.importances_mean, yerr=imps.importances_std)
    plt.bar_label(C, labels)
    plt.show()


def graphviz_feature(feature):
    G = graphviz.Graph()
    i = 0

    def C():
        nonlocal i
        i += 1
        return str(i)

    def rec(t):
        id = C()
        match t:
            case (f, *x):
                G.node(id, label=f)
                G.edges((id, rec(x_)) for x_ in x)
            case x:
                G.node(id, label=x)
        return id

    rec(feature)
    return G
