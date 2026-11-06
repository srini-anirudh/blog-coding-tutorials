"""Convolution from scratch: the sliding dot product, its geometry and its cost.

Everything here is checked against torch.nn.functional in the tests.
"""

import torch
import torch.nn.functional as F


def out_size(n_in, k, s=1, p=0, d=1):
    """floor((n + 2p - d(k-1) - 1) / s) + 1. With d = 1 this is the post's floor((n + 2p - k) / s) + 1."""
    return (n_in + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d_loops(x, w, b=None, stride=1, padding=0):
    """The post's first formula, one multiply-accumulate at a time.

    x: (C_in, H, W), w: (C_out, C_in, k, k). Cross-correlation, like every framework.
    Slow on purpose. Use it on tiny inputs only.
    """
    c_in, h, wd = x.shape
    c_out, _, k, _ = w.shape
    x = F.pad(x, (padding, padding, padding, padding))
    ho, wo = out_size(h, k, stride, padding), out_size(wd, k, stride, padding)
    y = torch.zeros(c_out, ho, wo, dtype=x.dtype)
    for o in range(c_out):
        for i in range(ho):
            for j in range(wo):
                acc = 0.0
                for c in range(c_in):
                    for u in range(k):
                        for v in range(k):
                            acc += w[o, c, u, v] * x[c, i * stride + u, j * stride + v]
                y[o, i, j] = acc + (b[o] if b is not None else 0.0)
    return y


def conv2d_im2col(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution as one matrix multiply per group (im2col).

    x: (N, C_in, H, W), w: (C_out, C_in / groups, k, k). Every patch becomes a column, so the layer is
    (C_out, C_in k k) @ (C_in k k, H_out W_out). Groups make the weight block-diagonal: each group of
    output channels only multiplies its own slice of the patch columns.
    """
    n, c_in, h, wd = x.shape
    c_out, cg, k, _ = w.shape
    assert c_in == cg * groups and c_out % groups == 0
    ho, wo = out_size(h, k, stride, padding, dilation), out_size(wd, k, stride, padding, dilation)
    cols = F.unfold(x, k, dilation=dilation, padding=padding, stride=stride)  # (N, C_in k k, L)
    cols = cols.view(n, groups, cg * k * k, ho * wo)
    wg = w.view(groups, c_out // groups, cg * k * k)
    y = torch.einsum("gok,ngkl->ngol", wg, cols).reshape(n, c_out, ho, wo)
    if b is not None:
        y = y + b.view(1, -1, 1, 1)
    return y


def conv_transpose2d_scatter(x, w, stride=2, padding=0):
    """Transposed convolution as scatter: every input pixel stamps a weighted copy of the kernel.

    x: (N, C_in, H, W), w: (C_in, C_out, k, k) (PyTorch's layout for transposed conv).
    Output size is (H - 1) s - 2p + k. Overlapping stamps are what cause checkerboard artefacts
    when k is not a multiple of s.
    """
    n, c_in, h, wd = x.shape
    _, c_out, k, _ = w.shape
    hf, wf = (h - 1) * stride + k, (wd - 1) * stride + k
    y = torch.zeros(n, c_out, hf, wf, dtype=x.dtype)
    stamp = torch.einsum("nchw,cokl->nohwkl", x, w)
    for u in range(k):
        for v in range(k):
            y[:, :, u:u + stride * h:stride, v:v + stride * wd:stride] += stamp[..., u, v]
    return y[:, :, padding:hf - padding, padding:wf - padding]


# ---------------------------------------------------------------- cost formulas

def conv_params(k, c_in, c_out, groups=1, bias=False, kt=1):
    """k k C_in C_out / g weights (times kt for 3-D), plus C_out biases."""
    return kt * k * k * c_in * c_out // groups + (c_out if bias else 0)


def conv_macs(k, c_in, c_out, out_positions, groups=1, kt=1):
    """One multiply-accumulate per weight per output position."""
    return kt * k * k * c_in * c_out // groups * out_positions


def family_table(hw=56, c=128, k=3):
    """The post's 'convolution family' table at H = W = 56, C_in = C_out = 128, k = 3. No biases."""
    pos = hw * hw
    rows = [
        ("Standard 3x3", conv_params(k, c, c), conv_macs(k, c, c, pos)),
        ("1x1 (pointwise)", conv_params(1, c, c), conv_macs(1, c, c, pos)),
        ("Depthwise 3x3", conv_params(k, c, c, groups=c), conv_macs(k, c, c, pos, groups=c)),
        ("Depthwise separable", conv_params(k, c, c, groups=c) + conv_params(1, c, c),
         conv_macs(k, c, c, pos, groups=c) + conv_macs(1, c, c, pos)),
        ("Grouped 3x3, g=32", conv_params(k, c, c, groups=32), conv_macs(k, c, c, pos, groups=32)),
        ("Dilated 3x3, d=2", conv_params(k, c, c), conv_macs(k, c, c, pos)),
        # A stride-2 transposed conv maps 56x56 to 112x112; count one MAC per weight per output position.
        ("Transposed 3x3, s=2", conv_params(k, c, c), conv_macs(k, c, c, (2 * hw) ** 2)),
    ]
    return rows


def separable_ratio(k, c_out):
    """Cost of depthwise-separable relative to standard: 1/C_out + 1/k^2."""
    return 1 / c_out + 1 / k ** 2


def stacked_vs_single(c, n_stack=3, k_small=3):
    """Params of n stacked k_small x k_small convs versus one conv with the same receptive field."""
    k_big = n_stack * (k_small - 1) + 1
    return n_stack * k_small ** 2 * c * c, k_big ** 2 * c * c, k_big


def unpadded_shrink(n, k=3, layers=16):
    for _ in range(layers):
        n = out_size(n, k)
    return n


def rotate_kernel(w):
    return torch.rot90(w, 1, dims=(-2, -1))


KERNELS = {
    "identity": torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]]),
    "sobel_x": torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "sobel_y": torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]),
    "laplacian": torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]]),
    "box_blur": torch.full((3, 3), 1 / 9),
    "gaussian": torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    "sharpen": torch.tensor([[0., -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "emboss": torch.tensor([[-2., -1, 0], [-1, 1, 1], [0, 1, 2]]),
}


def cosine_to_patch(w, patch):
    """A filter's response is a dot product, so it is cosine similarity times the two norms."""
    return float((w * patch).sum() / (w.norm() * patch.norm() + 1e-12))

