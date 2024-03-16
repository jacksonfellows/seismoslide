import seisbench.data
from matplotlib import pyplot as plt

from normalize import normalize

# Shouldn't need to specific component_order.
train = seisbench.data.WaveformDataset("pnw_all", component_order="Z").train()


def plot_sample_waveforms():
    N = 4
    cats = train.metadata.source_type.unique()
    fig, axs = plt.subplots(
        nrows=len(cats), ncols=N, layout="tight", sharex=True, sharey=True
    )
    for cati, cat in enumerate(cats):
        axs[cati, 0].set_ylabel(cat)
        i = train.metadata[train.metadata.source_type == cat].index
        print(cati, cat, i)
        for n in range(N):
            waveform = train.get_waveforms(i[n])[0]
            waveform = normalize(waveform)
            arrival = train.metadata.iloc[i[n]].get("trace_P_arrival_sample")
            axs[cati, n].plot(waveform, color="k", linewidth=0.5)
            if arrival:
                axs[cati, n].vlines(arrival, -8, 8, color="red", linewidth=0.5)
    plt.show()
