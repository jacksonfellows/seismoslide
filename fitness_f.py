import numpy as np
from matplotlib import pyplot as plt

from evolve import load_evolver, score_model


def f(x, window_len):
    a = (
        np.sum(np.lib.stride_tricks.sliding_window_view(x, window_len), axis=-1)
        / window_len
        - 0.5
    )
    return np.mean(np.abs(a))


def lets_see(random_fs=None):
    train = load_evolver("train")
    valid = load_evolver("valid")
    if random_fs is None:
        random_fs = [train.random_feature(5, 0.2) for _ in range(5)]
    # random_fs.extend([("kurtosis", ("abs", "Z")), ("kurtosis", ("abs_fft", "Z"))])
    print(f"{random_fs=}")
    imps = score_model(
        train, valid, random_fs, n_jobs=-1, calculate_permutation_importances=True
    )
    print(f"mean imp.={imps.importances_mean}")
    X = train.eval_features(random_fs)
    # ii = np.random.permutation(np.arange(X.shape[0]))
    X_ = X
    y_ = train.y
    # imps_m = imps.importances_mean - imps.importances_mean.min()
    plt.plot(imps.importances_mean, "x")
    for l in [10, 100, 500, 1000]:
        f_scores = np.array([f(y_[np.argsort(X_[:, i])], l) for i in range(X.shape[1])])
        # print(f"{f_scores=}")
        # f_scores -= f_scores.min()
        plt.plot(f_scores, "o", label=f"window len={l}")
    plt.legend()
    plt.show()
