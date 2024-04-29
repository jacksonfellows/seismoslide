import torch
from matplotlib import pyplot as plt

import train
from normalize import normalize

model = torch.load("model1.pt")

test = train.dataset.test()


def plot_input():
    X, m = test.get_sample(3)
    print(m)
    X = X[0, :6000]
    fig, ax = plt.subplots(figsize=(5, 1), layout="tight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.yticks([])
    plt.xlim(0, 6000)
    plt.plot(X, color=(19 / 255, 144 / 255, 19 / 255), linewidth=0.5)
    plt.savefig("sample_su_input.svg")


def plot_output():
    X, m = test.get_sample(3)
    X = X[None, :, :6000]
    X = torch.tensor(normalize(X), dtype=torch.float32)
    y = model(X).detach().numpy()

    fig, axs = plt.subplots(nrows=4, sharex=True, layout="tight", figsize=(5, 4))
    labels = ["Earthquake", "Explosion", "Surface event", "Noise"]
    axs[0].set_xlim(0, 6000)
    for i in range(4):
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["left"].set_visible(False)
        # axs[i].set_xticks([])
        axs[i].set_yticks([0, 1])
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].plot(y[0, i], color=(126 / 255, 97 / 255, 194 / 255), linewidth=1)
        axs[i].set_ylabel(labels[i])

    axs[-1].set_xticks(range(0, 6000 + 1, 1000))
    plt.savefig("sample_su_output.svg")
