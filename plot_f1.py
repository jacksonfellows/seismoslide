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
    fig, ax = plt.subplots()
    n = dict(
        mean_f1="Mean", eq_f1="Earthquake", ex_f1="Explosion", su_f1="Surface Event"
    )
    plt.title("F1 Scores on Test Dataset")
    ax.set_xticklabels([n[c] for c in f1_scores.columns])
    plt.ylabel("F1 Score")
    plt.xlabel("Event Class")
    plt.boxplot(f1_scores)
    plt.show()
