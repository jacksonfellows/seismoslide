import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Blur pooling adapted from https://github.com/adobe/antialiased-cnns.
def get_pad_layer_1d(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad1d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class BlurPool1D(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2, pad_off=0, pad_type="zero"):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(math.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        # Binomial coefficients.
        a = [
            math.comb(filt_size - 1, n) / 2 ** (filt_size - 1) for n in range(filt_size)
        ]

        filt = torch.Tensor(a)
        self.register_buffer("filt", filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride]
        else:
            return F.conv1d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


class PhaseNetBlur(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        classes=3,
        sampling_rate=100,
        depth=5,
        kernel_size=7,
        stride=4,
        filters_root=8,
        blur_kernel_size=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters_root = filters_root
        self.activation = torch.relu

        self.blur_kernel_size = blur_kernel_size

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters, filters, self.kernel_size, padding=padding, bias=False
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)
                blurpool = BlurPool1D(
                    filters, filt_size=self.blur_kernel_size, stride=self.stride
                )

            self.down_branch.append(
                nn.ModuleList([conv_same, bn1, conv_down, bn2, blurpool])
            )

        for i in range(self.depth - 1):
            filters = int(2 ** (self.depth - 2 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            blurpool = BlurPool1D(filters, filt_size=self.blur_kernel_size, stride=1)

            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(
                nn.ModuleList([conv_up, bn1, blurpool, conv_same, bn2])
            )

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2, blurpool) in enumerate(
            self.down_branch
        ):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = blurpool(self.activation(bn2(conv_down(x))))

        for i, ((conv_up, bn1, blurpool, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = blurpool(self.activation(bn1(conv_up(x))))
            # x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        if offset >= 0:
            x = x[:, :, offset : offset + skip.shape[-1]]
        elif offset < 0:
            x = F.pad(x, (skip.shape[-1] - x.shape[-1], 0), "constant", 0)

        return torch.cat([skip, x], dim=1)
