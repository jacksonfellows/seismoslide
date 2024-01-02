from matplotlib import pyplot as plt

from mydataset import MyDataset
from normalize import normalize

train = MyDataset("./pnw_splits/train/")


def plot_sample_waveforms():
    N = 4
    cats = train.metadata.source_type.unique()
    fig, axs = plt.subplots(
        nrows=len(cats), ncols=N, layout="tight", sharex=True, sharey=True
    )
    for cati, cat in enumerate(cats):
        axs[cati, 0].set_ylabel(cat)
        i = train.metadata[train.metadata.source_type == cat].index
        for n in range(N):
            waveform = train[i[n]]
            waveform = normalize(waveform)
            arrival = train.metadata.iloc[i[n]].get("trace_P_arrival_sample")
            axs[cati, n].plot(waveform)
            if arrival:
                axs[cati, n].vlines(arrival, -8, 8, color="red")
    plt.show()
