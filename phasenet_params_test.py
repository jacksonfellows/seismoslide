import itertools

import torch

from my_phasenet import PhaseNet

X = torch.rand((1, 1, 3001))


def make_model(depth, kernel_size, stride):
    return PhaseNet(
        in_channels=1,
        classes=4,
        phases=[
            "earthquake",
            "explosion",
            "surface event",
            "noise",
        ],  # class names,
        sampling_rate=100,
        depth=depth,
        kernel_size=kernel_size,
        stride=stride,
    )


def t():
    n_failed = 0
    n_tot = 0
    for depth, kernel_size, stride in itertools.product(
        range(3, 7 + 1), range(3, 11 + 1), (2, 4, 8)
    ):
        try:
            m = make_model(depth, kernel_size, stride)
            m(X)
            # print(f"succeeded {depth=} {kernel_size=} {stride=}")
        except:
            print(f"failed {depth=} {kernel_size=} {stride=}")
            n_failed += 1
        n_tot += 1
    print(f"failures: {n_failed}/{n_tot}")
