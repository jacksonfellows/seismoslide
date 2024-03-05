import sys

import numpy as np
import scipy
import seisbench.data
import sounddevice as sd

from normalize import normalize

pnw = seisbench.data.PNW()
pnw_exotic = seisbench.data.PNWExotic()

if __name__ == "__main__":
    speedup = int(sys.argv[1])
    dataset = dict(pnw=pnw, pnw_exotic=pnw_exotic)[sys.argv[2]]
    i = int(sys.argv[3])
    x, metadata = dataset.get_sample(i)
    x = x[0]  # Z component
    x = normalize(x)
    x /= np.abs(x).max()
    print(metadata)
    sd.play(x, speedup * 100, blocking=True)
    # scipy.io.wavfile.write("test.wav", speedup * 100, x)
