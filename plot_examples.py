import matplotlib.pyplot as plt
import numpy as np
import seisbench.data
import torch
from matplotlib.patches import Patch

from normalize import normalize

test = seisbench.data.WaveformDataset("pnw_all", component_order="Z").test()

model = torch.load("../pytorch-seismic-inference/model1.pt")


def plot(seed=123):
    rng = np.random.default_rng(seed)
    N = 4
    main_fig = plt.figure(
        layout="constrained",
        figsize=[8.5, 8.5],
    )
    fig, legend_fig = main_fig.subfigures(nrows=2, ncols=1, height_ratios=[100, 9])
    axs = fig.subplots(
        nrows=2 * N,
        ncols=3,
        sharex=True,
        sharey="row",
        height_ratios=[1, 0.4] * N,
    )

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # for n in range(N):
    #     for cati in range(3):
    #         axs[2 * n + 1, cati].spines["bottom"].set_visible(True)

    for n in range(N):
        axs[2 * n, 0].set_xlim(0, 60)
        axs[2 * n, 0].set_ylabel("Amplitude")
        axs[2 * n + 1, 0].set_ylim(-0.1, 1.1)
        axs[2 * n + 1, 0].set_ylabel("Probability")

    arrival_colors = ["red", "blue", "green", "grey"]
    arrival_labels = [
        "P(Earthquake arrival)",
        "P(Explosion arrival)",
        "P(Surface event arrival)",
        "P(Noise)",
    ]
    cats = ["earthquake", "explosion", "surface event"]
    pick_labels = [
        "Earthquake arrival pick",
        "Explosion arrival pick",
        "Surface event arrival pick",
    ]
    x = np.arange(6000) / 100
    for cati, cat in enumerate(cats):
        axs[0, cati].set_title(cat.capitalize())
        axs[-1, cati].set_xlabel("Time [s]")
        # Hacky
        i = (
            test.metadata[test.metadata.source_type == cat].index
            - test.metadata.index[0]
        )
        for n in range(N):
            waveform = normalize(test.get_waveforms(i[n + 3]))[0]
            start = int(rng.random() * 1200)
            waveform = waveform[start : 6000 + start]
            assert len(waveform) == 6000
            arrival = test.metadata.iloc[i[n + 3]].get("trace_P_arrival_sample")
            arrival -= start
            axs[2 * n, cati].plot(x, waveform, color="k", linewidth=0.75)
            if arrival:
                M = 17
                axs[2 * n, cati].vlines(
                    arrival / 100,
                    -M,
                    +M,
                    color=arrival_colors[cati],
                    linewidth=0.75,
                    linestyle="--",
                    label=pick_labels[cati],
                )

            X = torch.tensor(waveform.reshape(1, 1, -1), dtype=torch.float32)
            # Already normalized.
            y = model(X).detach().numpy()

            for ai in reversed(list(range(len(arrival_labels)))):
                axs[2 * n + 1, cati].plot(
                    x,
                    y[0, ai],
                    color=arrival_colors[ai],
                    label=arrival_labels[ai],
                    linewidth=0.75,
                )

    handles, labels = axs[1, 0].get_legend_handles_labels()
    handles = list(reversed(handles))
    labels = list(reversed(labels))
    for cati in range(3):
        h, l = axs[0, cati].get_legend_handles_labels()
        handles += h
        labels += l
    I = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[i] for i in I]
    labels = [labels[i] for i in I]
    legend_fig.legend(handles=handles, labels=labels, ncols=4, loc="center")

    plt.savefig("examples.pdf")
