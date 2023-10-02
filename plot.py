import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evolve import default_evolver

df1 = pd.read_csv("n_random_features_vs_model_score_cleanup.csv")
df2 = pd.read_csv("n_random_features_vs_model_score_2.csv")


def p(df):
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


def plot_features(X, y, c1, c2):
    X0 = X[y == 0]
    X1 = X[y == 1]
    # TODO: Make sure labels match.
    plt.scatter(
        X0[:, c1], X0[:, c2], color="red", facecolors="none", label="earthquake"
    )
    plt.scatter(
        X1[:, c1], X1[:, c2], color="blue", facecolors="none", label="surface event"
    )
    plt.legend()
    plt.show()


def plot_waveform(W, i, components=None):
    sampling_rate = 100
    ws = W[i]
    x = np.arange(ws.shape[1]) / sampling_rate
    if components is None:
        components = "ZNE"
    for i, l in enumerate("ZNE"):
        if l in components:
            plt.plot(x, ws[i], label=l)
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
