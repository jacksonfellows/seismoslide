import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_f1():
    f1_scores = pd.DataFrame(columns=["mean_f1", "eq_f1", "ex_f1", "su_f1"])
    for col in f1_scores.columns:
        df = pd.read_csv(f"test_f1_scores/{col}.csv")
        f1_scores[col] = df[df.columns[1]]
    return f1_scores


f1_scores = load_f1()


def plot_f1():
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="tight")
    n = dict(
        mean_f1="Mean of all classes",
        eq_f1="Earthquake",
        ex_f1="Explosion",
        su_f1="Surface event",
    )
    plt.title("Test dataset (100 runs)")
    ax.set_xticklabels([n[c] for c in f1_scores.columns])
    plt.ylabel("F1 score")
    plt.ylim(0.9, 1.0)
    plt.yticks(np.arange(0.9, 1.01, 0.01))
    plt.xlabel("Event class")
    plt.boxplot(f1_scores, medianprops=dict(color="r"))
    plt.savefig("f1_scores.png", dpi=300)
