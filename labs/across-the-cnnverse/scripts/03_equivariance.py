"""Why convolution wins: weight sharing, translation equivariance, and where equivariance breaks.

1. Parameter count of a dense layer against a conv layer on the same image.
2. f(shift(x)) == shift(f(x)) for a stride-1 conv; what stride 2, max pooling and global pooling do.
3. A dense layer can express any convolution: build the conv as a dense matrix and compare.
4. A 2-D conv over frames stacked as channels is not shift-equivariant in time; a 3-D conv is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnnverse.report import Report, set_threads


def shift(x, dx):
    """Circular shift along the width axis. Circular padding makes the comparison exact at the borders."""
    return torch.roll(x, dx, dims=-1)


def equivariance_error(f, x, dx, out_shift):
    a, b = f(shift(x, dx)), shift(f(x), out_shift)
    return float((a - b).abs().max() / b.abs().max())


def conv_as_dense(w, h, wd):
    """The (C_out H W) x (C_in H W) matrix that a 'same'-padded stride-1 conv computes."""
    c_in = w.shape[1]
    eye = torch.eye(c_in * h * wd).view(-1, c_in, h, wd)
    cols = F.conv2d(eye, w, padding=w.shape[-1] // 2)
    return cols.flatten(1).T


def main():
    set_threads()
    R = Report("03_equivariance.txt")
    g = torch.Generator().manual_seed(0)

    R("\n== The parameter cost of not sharing ==")
    dense = 224 * 224 * 3 * 1000 + 1000
    R(f"dense layer, 224x224x3 image -> 1000 units: {dense:,} parameters ({dense / 1e6:.1f} M)")
    R(f"3x3 conv, 3 -> 1000 channels: {3 * 3 * 3 * 1000 + 1000:,} parameters, at every one of 224x224 positions")
    w_small = torch.randn(4, 3, 3, 3, generator=g)
    M = conv_as_dense(w_small, 8, 8)
    nz = (M != 0).sum().item()
    R(f"the 8x8 conv layer (3 -> 4 ch) written as a dense matrix: shape {tuple(M.shape)}, {M.numel():,} entries,"
      f" {nz:,} nonzero, {w_small.numel()} distinct values")
    x = torch.randn(1, 3, 8, 8, generator=g)
    err = (M @ x.flatten() - F.conv2d(x, w_small, padding=1).flatten()).abs().max()
    R(f"dense matrix @ x vs F.conv2d: max abs error {err:.2e} (convolution is a strict subset of dense layers)")

    R("\n== Translation equivariance, measured (relative max error; 0 means exact) ==")
    x = torch.randn(1, 3, 32, 32, generator=g)
    conv1 = nn.Conv2d(3, 8, 3, padding=1, padding_mode="circular", bias=False)
    conv2 = nn.Conv2d(3, 8, 3, stride=2, padding=1, padding_mode="circular", bias=False)
    pool = nn.MaxPool2d(2)
    with torch.no_grad():
        R(f"{'operation':>40} {'shift 1':>9} {'shift 2':>9} {'shift 4':>9}")
        rows = [
            ("stride-1 conv, output shifted by dx", conv1, lambda d: d),
            ("stride-2 conv, output shifted by dx/2", conv2, lambda d: d // 2),
            ("2x2 max pool, output shifted by dx/2", pool, lambda d: d // 2),
            ("conv -> ReLU -> conv (stride 1)", nn.Sequential(conv1, nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1,
             padding_mode="circular")), lambda d: d),
        ]
        for name, f, osh in rows:
            errs = [equivariance_error(f, x, d, osh(d)) for d in (1, 2, 4)]
            R(f"{name:>40} " + " ".join(f"{e:9.3f}" for e in errs))
        R("a stride-2 layer is equivariant only to even shifts: an odd shift lands between its samples (aliasing)")
        feat = torch.relu(conv1(x))
        sub = lambda t: t[..., ::2, ::2]
        same_pool = float((pool(shift(feat, 1)) == pool(feat)).float().mean())
        same_sub = float((sub(shift(feat, 1)) == sub(feat)).float().mean())
        R(f"after conv + ReLU, share of outputs unchanged by a 1-px shift: 2x2 max pool {same_pool:.2f},"
          f" plain stride-2 subsample {same_sub:.2f} (max pooling buys a little local invariance;"
          f" the subsample's matches are only ReLU zeros)")
        gap = lambda t: F.adaptive_avg_pool2d(conv1(t), 1)
        inv = [float((gap(shift(x, d)) - gap(x)).abs().max()) for d in (1, 3, 7)]
        R(f"global average pool after a conv: change in output for shifts 1, 3, 7 = {', '.join(f'{v:.1e}' for v in inv)}"
          f" (equivariance turned into invariance)")
        lin = nn.Linear(3 * 32 * 32, 8)
        dl = float((lin(shift(x, 1).flatten(1)) - lin(x.flatten(1))).abs().max() / lin(x.flatten(1)).abs().max())
        R(f"a dense layer on the same image: relative change for a 1-pixel shift = {dl:.2f} (no structure)")

    R("\n== Motion needs a swept time axis ==")
    frames = torch.zeros(1, 1, 8, 16, 16)
    for t in range(8):  # a bright bar moving right one pixel per frame
        frames[0, 0, t, :, 4 + t] = 1.0
    c3 = nn.Conv3d(1, 1, (3, 3, 3), padding=1, bias=False)
    c2 = nn.Conv2d(8, 1, 3, padding=1, bias=False)  # all 8 frames as input channels
    with torch.no_grad():
        tshift = torch.roll(frames, 2, dims=2)
        y3a, y3b = c3(tshift), torch.roll(c3(frames), 2, dims=2)
        inner = slice(3, 7)  # away from the zero-padded ends of the time axis
        e3 = float((y3a[:, :, inner] - y3b[:, :, inner]).abs().max())
        y2a, y2b = c2(tshift[:, 0]), c2(frames[:, 0])
        e2 = float((y2a - y2b).abs().max() / y2b.abs().max())
        R(f"3-D conv: shifting the clip by 2 frames shifts the output by 2 frames, max error {e3:.1e}")
        R(f"2-D conv over frames-as-channels: the same time shift changes the single output by {e2:.2f} (relative);"
          f" each frame has its own weight slot, so the filter is a fixed template, not a moving-edge detector")
    R.close()


if __name__ == "__main__":
    main()
