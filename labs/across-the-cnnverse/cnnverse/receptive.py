"""Receptive fields: the post's recursion, checked against what gradients actually reach."""

import torch
import torch.nn as nn


def rf_recursion(layers):
    """layers: list of (k, s, d). Returns [(jump, rf)] after each layer.

    j_n = j_{n-1} s_n,   r_n = r_{n-1} + (k_n - 1) d_n j_{n-1},   j_0 = r_0 = 1.
    """
    j, r, out = 1, 1, []
    for k, s, d in layers:
        r = r + (k - 1) * d * j
        j = j * s
        out.append((j, r))
    return out


def conv_stack(layers, channels=1, act=False):
    """Plain conv stack with padding that keeps the centre aligned. Weights are 1, so every path counts."""
    mods = []
    for k, s, d in layers:
        conv = nn.Conv2d(channels, channels, k, s, padding=d * (k - 1) // 2, dilation=d, bias=False)
        nn.init.constant_(conv.weight, 1.0)
        mods.append(conv)
        if act:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def gradient_map(model, size, channels=1, weights="ones", seed=0):
    """|d y_centre / d x| over the input, for the centre output unit of the last layer.

    weights="ones" gives the theoretical footprint (every path is nonzero).
    weights="random" gives the effective receptive field of Luo et al. (2017).
    """
    g = torch.Generator().manual_seed(seed)
    if weights == "random":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) / (m.weight[0].numel() ** 0.5))
    x = torch.randn(1, channels, size, size, generator=g) if weights == "random" else torch.ones(1, channels, size, size)
    x.requires_grad_(True)
    y = model(x)
    c = y.shape[-1] // 2
    y[0, :, c, c].sum().backward()
    return x.grad[0].abs().sum(0)


def support_width(grad_map, tol=0.0):
    """Width of the nonzero region along the centre row."""
    row = grad_map[grad_map.shape[0] // 2]
    nz = (row > tol).nonzero().flatten()
    return int(nz[-1] - nz[0] + 1) if len(nz) else 0


def mass_within(grad_map, half_width):
    """Fraction of the gradient magnitude that falls in the central (2 half_width + 1)^2 square."""
    c = grad_map.shape[0] // 2
    tot = grad_map.sum()
    inner = grad_map[c - half_width:c + half_width + 1, c - half_width:c + half_width + 1].sum()
    return float(inner / tot)


# The four stacking strategies of Figure 04, each as (k, s, d) per layer.
STRATEGIES = {
    "3x3, stride 1, x8": [(3, 1, 1)] * 8,
    "3x3, stride 2 every 2nd layer, x8": [(3, 2 if i % 2 == 1 else 1, 1) for i in range(8)],
    "3x3, dilation 1,2,4,8 (x2)": [(3, 1, d) for d in (1, 2, 4, 8, 1, 2, 4, 8)],
    "7x7, stride 1, x3": [(7, 1, 1)] * 3,
}
