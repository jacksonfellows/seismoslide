import numpy as np
import seisbench.data
from matplotlib import pyplot as plt

pnw_exotic = seisbench.data.PNWExotic()


def metadata_stats():
    m = pnw_exotic.metadata
    source_types = m.source_type.unique()
    fig, axs = plt.subplots(nrows=len(source_types), sharex=True)
    for ax, source_type in zip(axs, source_types):
        e = m[m.source_type == source_type]
        g = e.groupby("event_id")
        ax.set_ylabel("# of events")
        ax.set_title(f"{source_type} ({len(e.event_id.unique())} unique events)")
        g.station_code.count().hist(ax=ax)
    axs[-1].set_xlabel("# of stations")
    plt.show()


def plot_waveform(idx, components=None):
    sampling_rate = 100
    ws = pnw_exotic.get_waveforms(idx, sampling_rate=sampling_rate)
    x = np.arange(ws.shape[1]) / sampling_rate
    if components is None:
        components = pnw_exotic.component_order
    for i, l in enumerate(pnw_exotic.component_order):
        if l in components:
            plt.plot(x, ws[i], label=l)
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
