"""The post's nine networks, plus ResNeXt-50, SE-ResNet-50 and the CIFAR plain/residual pair.

Each is written from the paper's layer table. They are untrained: the point is the shapes, the parameter
counts and the cost, which are read off the modules by `cnnverse.profile`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(c_in, c_out, k, s=1, groups=1, act=nn.ReLU):
    layers = [nn.Conv2d(c_in, c_out, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ LeNet-5 (1998)

LENET_C3_TABLE = [  # LeCun et al. 1998, Table I: which of the 6 S2 maps feed each of the 16 C3 maps
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 0), (5, 0, 1),
    (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 0), (4, 5, 0, 1), (5, 0, 1, 2),
    (0, 1, 3, 4), (1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 2, 3, 4, 5),
]


class MaskedConv2d(nn.Conv2d):
    """A conv whose weight is multiplied by a fixed 0/1 connection table."""

    def __init__(self, c_in, c_out, k, table):
        super().__init__(c_in, c_out, k)
        mask = torch.zeros(c_out, c_in, k, k)
        for o, ins in enumerate(table):
            mask[o, list(ins)] = 1
        self.register_buffer("mask", mask)
        self.n_weights = int(mask.sum())   # a Python int, so it survives a move to the meta device

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias)


class Subsample(nn.Module):
    """LeNet's trainable subsampling: 2x2 average, then a per-map scale and bias, then a squashing function."""

    def __init__(self, c):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        return torch.tanh(F.avg_pool2d(x, 2) * 4 * self.scale + self.bias)


def lenet5(paper=False):
    """paper=False: the common modern form (dense C3, plain pooling, linear output) = 61,706 parameters.
    paper=True: the 1998 form, with the sparse C3 table and trainable subsampling."""
    if paper:
        c3, s2, s4 = MaskedConv2d(6, 16, 5, LENET_C3_TABLE), Subsample(6), Subsample(16)
        out = nn.Linear(84, 10, bias=False)  # stands in for the RBF output layer: 84 x 10 centres, no bias
    else:
        c3, s2, s4 = nn.Conv2d(6, 16, 5), nn.AvgPool2d(2), nn.AvgPool2d(2)
        out = nn.Linear(84, 10)
    return nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), s2, c3, nn.Tanh(), s4,
                         nn.Conv2d(16, 120, 5), nn.Tanh(), nn.Flatten(), nn.Linear(120, 84), nn.Tanh(), out)


# ------------------------------------------------------------------ AlexNet (2012), VGG-16 (2014)

def alexnet(grouped=True):
    """The two-GPU layout: conv2, conv4 and conv5 are split in two groups that never talk."""
    g = 2 if grouped else 1
    features = nn.Sequential(
        nn.Conv2d(3, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2),                  # 227 -> 55 -> 27
        nn.Conv2d(96, 256, 5, padding=2, groups=g), nn.ReLU(), nn.MaxPool2d(3, 2),  # -> 13
        nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, 3, padding=1, groups=g), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1, groups=g), nn.ReLU(), nn.MaxPool2d(3, 2))  # -> 6
    head = nn.Sequential(nn.Flatten(), nn.Dropout(), nn.Linear(256 * 36, 4096), nn.ReLU(),
                         nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
    return nn.Sequential(features, head)


VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def vgg16():
    layers, c = [], 3
    for v in VGG16_CFG:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU()]
            c = v
    head = nn.Sequential(nn.Flatten(), nn.Linear(512 * 49, 4096), nn.ReLU(), nn.Dropout(),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, 1000))
    return nn.Sequential(nn.Sequential(*layers), head)


# ------------------------------------------------------------------ GoogLeNet (2014)

class Inception(nn.Module):
    def __init__(self, c_in, b1, b3r, b3, b5r, b5, bp, reduce=True):
        """reduce=False builds the naive module: no 1x1 reductions before the 3x3 and 5x5, no pool projection."""
        super().__init__()
        relu = nn.ReLU
        self.b1 = nn.Sequential(nn.Conv2d(c_in, b1, 1), relu())
        if reduce:
            self.b3 = nn.Sequential(nn.Conv2d(c_in, b3r, 1), relu(), nn.Conv2d(b3r, b3, 3, padding=1), relu())
            self.b5 = nn.Sequential(nn.Conv2d(c_in, b5r, 1), relu(), nn.Conv2d(b5r, b5, 5, padding=2), relu())
            self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(c_in, bp, 1), relu())
        else:
            self.b3 = nn.Sequential(nn.Conv2d(c_in, b3, 3, padding=1), relu())
            self.b5 = nn.Sequential(nn.Conv2d(c_in, b5, 5, padding=2), relu())
            self.bp = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)


GOOGLENET_CFG = {  # Szegedy et al. 2014, Table 1: #1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool proj
    "3a": (192, 64, 96, 128, 16, 32, 32), "3b": (256, 128, 128, 192, 32, 96, 64),
    "4a": (480, 192, 96, 208, 16, 48, 64), "4b": (512, 160, 112, 224, 24, 64, 64),
    "4c": (512, 128, 128, 256, 24, 64, 64), "4d": (512, 112, 144, 288, 32, 64, 64),
    "4e": (528, 256, 160, 320, 32, 128, 128), "5a": (832, 256, 160, 320, 32, 128, 128),
    "5b": (832, 384, 192, 384, 48, 128, 128),
}


def googlenet():
    inc = {k: Inception(*v) for k, v in GOOGLENET_CFG.items()}
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
        nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
        inc["3a"], inc["3b"], nn.MaxPool2d(3, 2, 1),
        inc["4a"], inc["4b"], inc["4c"], inc["4d"], inc["4e"], nn.MaxPool2d(3, 2, 1),
        inc["5a"], inc["5b"], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(1024, 1000))


# ------------------------------------------------------------------ ResNet-50, ResNeXt-50, SE-ResNet-50

class SE(nn.Module):
    """Squeeze-and-excitation: global pool, bottleneck MLP, sigmoid, rescale every channel."""

    def __init__(self, c, hidden, act=nn.ReLU):
        super().__init__()
        self.fc1, self.fc2, self.act = nn.Conv2d(c, hidden, 1), nn.Conv2d(hidden, c, 1), act()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(s))))


class Bottleneck(nn.Module):
    def __init__(self, c_in, width, c_out, stride, groups=1, se=False):
        super().__init__()
        self.f = nn.Sequential(conv_bn(c_in, width, 1), conv_bn(width, width, 3, stride, groups),
                               conv_bn(width, c_out, 1, act=None))
        self.se = SE(c_out, c_out // 16) if se else nn.Identity()
        self.short = nn.Identity() if stride == 1 and c_in == c_out else conv_bn(c_in, c_out, 1, stride, act=None)

    def forward(self, x):
        return F.relu(self.se(self.f(x)) + self.short(x))


def resnet50(groups=1, width_per_group=64, se=False, blocks=(3, 4, 6, 3)):
    """ResNet-50 (v1.5: the stride sits in the 3x3). groups=32, width_per_group=4 gives ResNeXt-50 32x4d."""
    layers, c_in = [conv_bn(3, 64, 7, 2), nn.MaxPool2d(3, 2, 1)], 64
    for i, n in enumerate(blocks):
        width, c_out = groups * width_per_group * 2 ** i, 256 * 2 ** i
        stage = [Bottleneck(c_in if j == 0 else c_out, width, c_out, 2 if (j == 0 and i > 0) else 1, groups, se)
                 for j in range(n)]
        layers.append(nn.Sequential(*stage))
        c_in = c_out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 1000)]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ DenseNet-121 (2016)

class DenseLayer(nn.Module):
    def __init__(self, c_in, growth, bn_size=4):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm2d(c_in), nn.ReLU(), nn.Conv2d(c_in, bn_size * growth, 1, bias=False),
                               nn.BatchNorm2d(bn_size * growth), nn.ReLU(),
                               nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.f(x)], 1)   # concatenate and keep, instead of add and forget


def densenet121(growth=32, blocks=(6, 12, 24, 16)):
    layers, c = [conv_bn(3, 64, 7, 2), nn.MaxPool2d(3, 2, 1)], 64
    for i, n in enumerate(blocks):
        layers.append(nn.Sequential(*[DenseLayer(c + j * growth, growth) for j in range(n)]))
        c += n * growth
        if i < len(blocks) - 1:  # transition: compress channels by half, then 2x2 average pool
            layers.append(nn.Sequential(nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c // 2, 1, bias=False),
                                        nn.AvgPool2d(2)))
            c //= 2
    layers += [nn.BatchNorm2d(c), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1000)]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ MobileNetV2 (2018), EfficientNet (2019)

class MBConv(nn.Module):
    """Inverted residual: narrow -> wide (1x1) -> depthwise kxk -> optional SE -> narrow (1x1, no activation)."""

    def __init__(self, c_in, c_out, expand, k, stride, se_ratio=0.0, act=nn.ReLU6):
        super().__init__()
        mid = c_in * expand
        layers = [] if expand == 1 else [conv_bn(c_in, mid, 1, act=act)]
        layers.append(conv_bn(mid, mid, k, stride, groups=mid, act=act))
        if se_ratio:
            layers.append(SE(mid, max(1, int(c_in * se_ratio)), act=act))
        layers.append(conv_bn(mid, c_out, 1, act=None))   # the linear bottleneck
        self.f = nn.Sequential(*layers)
        self.skip = stride == 1 and c_in == c_out

    def forward(self, x):
        return x + self.f(x) if self.skip else self.f(x)


def mobilenet_v2():
    cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    layers, c = [conv_bn(3, 32, 3, 2, act=nn.ReLU6)], 32
    for t, c_out, n, s in cfg:
        for j in range(n):
            layers.append(MBConv(c, c_out, t, 3, s if j == 0 else 1))
            c = c_out
    layers += [conv_bn(c, 1280, 1, act=nn.ReLU6), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, 1000)]
    return nn.Sequential(*layers)


EFFNET_B0 = [(1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
             (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]  # expand, k, stride, channels, repeats


def round_filters(c, width, divisor=8):
    c *= width
    new = max(divisor, int(c + divisor / 2) // divisor * divisor)
    return int(new + divisor) if new < 0.9 * c else int(new)


def efficientnet(width=1.0, depth=1.0):
    """EfficientNet-B0 at width = depth = 1. Other members scale channels by `width` and repeats by `depth`."""
    stem = round_filters(32, width)
    layers, c = [conv_bn(3, stem, 3, 2, act=nn.SiLU)], stem
    for e, k, s, c_out, n in EFFNET_B0:
        c_out = round_filters(c_out, width)
        for j in range(int(math.ceil(n * depth))):
            layers.append(MBConv(c, c_out, e, k, s if j == 0 else 1, se_ratio=0.25, act=nn.SiLU))
            c = c_out
    head = round_filters(1280, width)
    layers += [conv_bn(c, head, 1, act=nn.SiLU), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(head, 1000)]
    return nn.Sequential(*layers)


# The (width, depth, resolution) the EfficientNet authors released for B0-B7.
EFFNET_MEMBERS = {"B0": (1.0, 1.0, 224), "B1": (1.0, 1.1, 240), "B2": (1.1, 1.2, 260), "B3": (1.2, 1.4, 300),
                  "B4": (1.4, 1.8, 380), "B5": (1.6, 2.2, 456), "B6": (1.8, 2.6, 528), "B7": (2.0, 3.1, 600)}


# ------------------------------------------------------------------ ConvNeXt-T (2022)

class ChannelsLastLN(nn.LayerNorm):
    """LayerNorm over the channel axis of an (N, C, H, W) tensor."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise -> LN -> 1x1 up 4x (as Linear) -> GELU -> 1x1 down -> layer scale -> add."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.up, self.act, self.down = nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(1e-6 * torch.ones(c))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.gamma * self.down(self.act(self.up(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)


def convnext_tiny(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
    layers = [nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), ChannelsLastLN(dims[0]))]  # patchify stem
    for i, (n, c) in enumerate(zip(depths, dims)):
        if i > 0:  # separate downsampling layer
            layers.append(nn.Sequential(ChannelsLastLN(dims[i - 1]), nn.Conv2d(dims[i - 1], c, 2, 2)))
        layers.append(nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n)]))
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(dims[-1]), nn.Linear(dims[-1], 1000)]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ registry

# name -> (builder, input size, published params in millions, published GMACs, ImageNet top-1 from the post)
NINE = {
    "LeNet-5": (lenet5, (1, 32, 32), 0.061706, 0.0004, None),
    "AlexNet": (alexnet, (3, 227, 227), 61.0, 0.72, 57.1),
    "VGG-16": (vgg16, (3, 224, 224), 138.4, 15.5, 71.6),
    "GoogLeNet": (googlenet, (3, 224, 224), 7.0, 1.6, 69.8),
    "ResNet-50": (resnet50, (3, 224, 224), 25.6, 4.1, 76.1),
    "DenseNet-121": (densenet121, (3, 224, 224), 8.0, 2.9, 74.4),
    "MobileNetV2": (mobilenet_v2, (3, 224, 224), 3.5, 0.30, 72.0),
    "EfficientNet-B0": (efficientnet, (3, 224, 224), 5.3, 0.39, 77.1),
    "ConvNeXt-T": (convnext_tiny, (3, 224, 224), 28.6, 4.5, 82.1),
}

EXTRAS = {
    "AlexNet, ungrouped": (lambda: alexnet(grouped=False), (3, 227, 227), None, None, None),
    "LeNet-5, 1998 form": (lambda: lenet5(paper=True), (1, 32, 32), None, None, None),
    "ResNeXt-50 32x4d": (lambda: resnet50(groups=32, width_per_group=4), (3, 224, 224), 25.0, 4.2, 77.8),
    "SE-ResNet-50": (lambda: resnet50(se=True), (3, 224, 224), 28.1, 4.1, 77.6),
}


# ------------------------------------------------------------------ CIFAR plain vs residual (He et al. 2015, sec. 4.2)

class CifarBlock(nn.Module):
    """Two 3x3 conv-BN layers. With residual=True, an option-A shortcut (subsample + zero-pad channels),
    which has no parameters, so the plain and residual networks have exactly the same weights."""

    def __init__(self, c_in, c_out, stride, residual):
        super().__init__()
        self.f = nn.Sequential(conv_bn(c_in, c_out, 3, stride), conv_bn(c_out, c_out, 3, act=None))
        self.residual, self.stride, self.pad = residual, stride, c_out - c_in

    def forward(self, x):
        y = self.f(x)
        if self.residual:
            s = x[:, :, ::self.stride, ::self.stride] if self.stride > 1 else x
            y = y + F.pad(s, (0, 0, 0, 0, 0, self.pad))
        return F.relu(y)


def cifar_net(depth=20, residual=True, width=16, classes=10):
    """Depth 6n + 2: a stem conv, three stages of n blocks at width, 2 width, 4 width, then pool and linear."""
    n = (depth - 2) // 6
    layers, c = [conv_bn(3, width, 3)], width
    for i in range(3):
        for j in range(n):
            c_out = width * 2 ** i
            layers.append(CifarBlock(c, c_out, 2 if (j == 0 and i > 0) else 1, residual))
            c = c_out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, classes)]
    return nn.Sequential(*layers)
