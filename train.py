import numpy as np
import seisbench.generate as sbg
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import models
from mydataset import MyDataset
from normalize import normalize

train = MyDataset("./pnw_splits/train")
valid = MyDataset("./pnw_splits/valid")

CLASSES = ["noise", "earthquake", "explosion", "surface event"]

S = 2**6


def add_classif_output(state_dict):
    waveform, metadata = state_dict["X"]
    P_arrival_sample = metadata.get("trace_P_arrival_sample")
    noise_i = CLASSES.index("noise")
    assert len(waveform) % S == 0
    N = len(waveform) // S
    if P_arrival_sample is None:
        assert metadata["source_type"] == "noise"
        classes = [noise_i for _ in range(N)]
    else:
        target_i = CLASSES.index(metadata["source_type"])
        L = len(waveform)
        classes = [
            target_i if (n * L / N) <= P_arrival_sample < ((n + 1) * L / N) else noise_i
            for n in range(N)
        ]
    one_hot = F.one_hot(
        torch.Tensor(classes).to(torch.int64), num_classes=len(CLASSES)
    ).T.to(torch.float32)
    state_dict["y"] = (one_hot, None)  # Need to indicate empty metadata!


def my_normalize(state_dict):
    waveform, metadata = state_dict["X"]
    state_dict["X"] = normalize(waveform).astype("float32"), metadata


WINDOW_LEN = 3072


def make_generator(dataset):
    gen = sbg.GenericGenerator(dataset)
    gen.augmentation(sbg.RandomWindow(windowlen=WINDOW_LEN))
    # TODO: Explore normalization options.
    # gen.augmentation(
    #     sbg.Normalize(
    #         demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="std"
    #     )
    # )
    gen.augmentation(my_normalize)
    gen.augmentation(add_classif_output)
    return gen


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    n_chunks = len(dataloader)
    for i, d in enumerate(dataloader):
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")
        loss = loss_fn(model(d["X"]), d["y"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


THRESHOLD = 0.3


def get_predicted(logits):
    # logits.shape == (num_batch, num_classes, num_segments)
    # Assumes noise is first class. Returns max non-noise class if any
    # predicted non-noise probabilities are above THRESHOLD.
    probs = F.softmax(logits, dim=1)
    non_noise = probs[:, 1:, :]
    return torch.where(
        torch.any(non_noise > THRESHOLD, axis=1), non_noise.argmax(axis=1) + 1, 0
    )


def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    test_loss = 0
    n_chunks = len(dataloader)
    TP, FP, FN = np.zeros(3), np.zeros(3), np.zeros(3)
    for i, d in enumerate(dataloader):
        if i % 10 == 0:
            print(f"{i}/{n_chunks}")
        y = d["y"]
        y_pred_logits = model(d["X"])
        loss = loss_fn(y_pred_logits, y)
        test_loss += loss
        y_pred = get_predicted(y_pred_logits)
        y_actual = y.argmax(axis=1)
        for classi in range(1, len(CLASSES)):
            TP[classi - 1] += torch.sum((y_actual == classi) & (y_pred == classi))
            FP[classi - 1] += torch.sum((y_actual != classi) & (y_pred == classi))
            FN[classi - 1] += torch.sum((y_actual == classi) & (y_pred != classi))
    mean_loss = test_loss / n_chunks
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    print(f"{mean_loss=:0.5f}, {precision=}, {recall=}")
    return mean_loss


def plot_model(model, X, y):
    y = y.detach().numpy()
    y_logits = model(X)
    y_prob = F.softmax(y_logits, dim=1).detach().numpy()
    fig, axs = plt.subplots(nrows=2, ncols=X.shape[0], sharex=True, sharey="row")
    for i in range(X.shape[0]):
        axs[0, i].plot(X[i])
        for classi, (classl, color) in enumerate(
            zip(CLASSES, ["blue", "red", "yellow", "green"])
        ):
            axs[1, i].plot(
                y[i, classi].repeat(S),
                label=f"expected {classl}",
                color=color,
            )
            axs[1, i].plot(
                y_prob[i, classi].repeat(S),
                label=f"output {classl}",
                color=color,
                linestyle="dashed",
            )
    fig.legend(loc="upper left")
    plt.show()


def train_test_loop(model, train_loader, valid_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    d = next(iter(valid_loader))
    sample_X = d["X"][:4]
    sample_y = d["y"][:4]
    try:
        for epoch in range(epochs):
            print(f"{epoch=}")
            train_loop(train_loader, model, loss_fn, optimizer, epoch)
            test_loop(valid_loader, model, loss_fn, epoch)
            plot_model(model, sample_X, sample_y)
    except KeyboardInterrupt:
        pass
    torch.save(model, "classifier_01.pt")


BATCH_SIZE = 64
train_loader = DataLoader(make_generator(train), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(make_generator(valid), batch_size=BATCH_SIZE, shuffle=True)
