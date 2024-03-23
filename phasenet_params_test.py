import itertools

import torch

from my_phasenet import PhaseNet

X = torch.rand((1, 1, 3001))


def make_model(depth, kernel_size, stride, filters_root):
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
        filters_root=filters_root,
    )


def t():
    n_failed = 0
    n_tot = 0
    for depth, kernel_size, stride, filters_root in itertools.product(
        (5, 6), (5, 7, 9, 11), (4,), (4, 6, 8)
    ):
        try:
            m = make_model(depth, kernel_size, stride, filters_root)
            m(X)
            # print(f"succeeded {depth=} {kernel_size=} {stride=}")
        except:
            print(f"failed {depth=} {kernel_size=} {stride=} {filters_root=}")
            n_failed += 1
        n_tot += 1
    print(f"failures: {n_failed}/{n_tot}")
