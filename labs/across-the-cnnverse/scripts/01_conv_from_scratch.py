"""Part one of the post: the operation, what a filter sees, the geometry and the convolution family.

Checks the hand-written convolutions against torch, then recomputes the post's cost tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cnnverse import conv
from cnnverse.report import Report, set_threads


def main():
    set_threads()
    R = Report("01_conv_from_scratch.txt")
    g = torch.Generator().manual_seed(0)

    R("\n== The operation: a 3x3 filter on a 6x6 input, one multiply-accumulate at a time ==")
    x = torch.randn(1, 6, 6, generator=g)
    w = torch.randn(1, 1, 3, 3, generator=g)
    y_loop = conv.conv2d_loops(x, w)
    y_ref = F.conv2d(x[None], w)[0]
    R(f"output shape {tuple(y_loop.shape[1:])}: 16 dot products of 9 terms, sharing 9 weights")
    R(f"max |loops - F.conv2d| = {(y_loop - y_ref).abs().max():.2e}")
    flipped = F.conv2d(x[None], torch.flip(w, (-2, -1)))[0]
    R(f"true convolution (flipped kernel) differs from cross-correlation by {(flipped - y_ref).abs().max():.3f};"
      f" learning the flipped weights gives it back exactly")

    R("\n== im2col with stride, padding, dilation and groups, against F.conv2d ==")
    R(f"{'k':>2} {'s':>2} {'p':>2} {'d':>2} {'g':>3}  {'out':>9}  {'formula':>7}  max abs error")
    for k, s, p, d, groups in [(3, 1, 1, 1, 1), (3, 2, 1, 1, 1), (5, 1, 2, 1, 1), (3, 1, 2, 2, 1),
                               (3, 1, 1, 1, 4), (3, 2, 1, 1, 16), (7, 4, 3, 1, 1), (1, 1, 0, 1, 1)]:
        xb = torch.randn(2, 16, 17, 17, generator=g)
        wb = torch.randn(8 if groups != 16 else 16, 16 // groups, k, k, generator=g)
        a = conv.conv2d_im2col(xb, wb, stride=s, padding=p, dilation=d, groups=groups)
        b = F.conv2d(xb, wb, stride=s, padding=p, dilation=d, groups=groups)
        R(f"{k:>2} {s:>2} {p:>2} {d:>2} {groups:>3}  {str(tuple(a.shape[-2:])):>9}  {conv.out_size(17, k, s, p, d):>7}"
          f"  {(a - b).abs().max():.2e}")
    xt, wt = torch.randn(1, 4, 5, 5, generator=g), torch.randn(4, 3, 3, 3, generator=g)
    err = (conv.conv_transpose2d_scatter(xt, wt, 2, 1) - F.conv_transpose2d(xt, wt, stride=2, padding=1)).abs().max()
    R(f"transposed conv by scatter vs F.conv_transpose2d (k=3, s=2, p=1): max abs error {err:.2e}")
    ones = conv.conv_transpose2d_scatter(torch.ones(1, 1, 6, 6), torch.ones(1, 1, 3, 3), 2)[0, 0]
    R(f"k=3, s=2 transposed conv of a constant image: output values {sorted(set(ones[2:-2, 2:-2].flatten().tolist()))}"
      f" -> the checkerboard (k is not a multiple of s)")

    R("\n== What a filter sees ==")
    flat = torch.ones(1, 1, 8, 8)
    edge = torch.zeros(1, 1, 8, 8)
    edge[..., 4:] = 1.0
    R(f"{'kernel':>10} {'sum':>6} {'flat-region response':>21} {'edge response (max |y|)':>24}")
    for name, k in conv.KERNELS.items():
        yf = F.conv2d(flat, k[None, None])
        ye = F.conv2d(edge, k[None, None])
        R(f"{name:>10} {k.sum():6.2f} {yf.mean():21.3f} {ye.abs().max():24.3f}")
    R("kernels that sum to zero are blind to flat regions; blurs sum to one and pass them unchanged")
    sx = conv.KERNELS["sobel_x"]
    vert = F.conv2d(edge, sx[None, None]).abs().max()
    horiz = F.conv2d(edge.transpose(-1, -2), sx[None, None]).abs().max()
    R(f"sobel_x on a vertical edge {vert:.1f}, on a horizontal edge {horiz:.1f};"
      f" rotate it 90 degrees and it equals sobel_y up to sign: {torch.equal(conv.rotate_kernel(sx).abs(), conv.KERNELS['sobel_y'].abs())}")
    R(f"doubling every weight doubles the response: {torch.allclose(F.conv2d(edge, 2 * sx[None, None]), 2 * F.conv2d(edge, sx[None, None]))}")
    patch = edge[0, 0, 2:5, 3:6]
    R(f"template matching: cosine(sobel_x, sobel_x) = {conv.cosine_to_patch(sx, sx):.3f},"
      f" cosine(sobel_x, a step-edge patch) = {conv.cosine_to_patch(sx, patch):.3f},"
      f" cosine(sobel_x, the reversed edge) = {conv.cosine_to_patch(sx, 1 - patch):.3f}")

    R("\n== Geometry ==")
    R(f"224 px through 16 unpadded 3x3 layers -> {conv.unpadded_shrink(224)} px")
    R(f"stride 2 quarters the output positions: 56x56 -> {conv.out_size(56, 3, 2, 1)}x{conv.out_size(56, 3, 2, 1)}")
    fp = 2 * (3 - 1) + 1
    R(f"a 3x3 kernel with dilation 2 covers a {fp}x{fp} footprint with 9 weights")
    stacked, single, kb = conv.stacked_vs_single(64)
    R(f"three stacked 3x3 = 27C^2 = {stacked:,}; one {kb}x{kb} = 49C^2 = {single:,} at C=64;"
      f" {1 - stacked / single:.0%} fewer")

    R("\n== The convolution family at H=W=56, C_in=C_out=128, k=3 (post's table) ==")
    R(f"{'operator':>22} {'params':>10} {'MACs':>12} {'vs standard':>12}")
    rows = conv.family_table()
    base = rows[0][2]
    for name, p, m in rows:
        R(f"{name:>22} {p:>10,} {m / 1e6:>10.1f} M {m / base:>11.4f}x")
    R(f"depthwise separable / standard = 1/C_out + 1/k^2 = {conv.separable_ratio(3, 128):.4f}"
      f" -> {1 / conv.separable_ratio(3, 128):.1f}x cheaper")

    R("\n== 1D, 2D, 3D: one operator, more swept axes ==")
    m2 = conv.conv_macs(3, 64, 64, 56 * 56)
    m3 = conv.conv_macs(3, 64, 64, 16 * 56 * 56, kt=3)
    R(f"3x3x64->64 at 56^2: {m2 / 1e6:.1f} M MACs; 3x3x3 at 16 frames of 56^2: {m3 / 1e6:.1f} M MACs; ratio {m3 / m2:.0f}x")
    x1, w1 = torch.randn(1, 4, 50, generator=g), torch.randn(6, 4, 5, generator=g)
    x3, w3 = torch.randn(1, 4, 6, 9, 9, generator=g), torch.randn(6, 4, 3, 3, 3, generator=g)
    R(f"Conv1d output {tuple(F.conv1d(x1, w1).shape[1:])}, params k*C_in*C_out = {w1.numel()}")
    R(f"Conv3d output {tuple(F.conv3d(x3, w3).shape[1:])}, params kt*k^2*C_in*C_out = {w3.numel()}")
    R.close()


if __name__ == "__main__":
    main()
