"""Section 08: position on a 2D patch grid.

1. Flattening a grid to 1D ids makes the end of one row the neighbour of the start of the next.
2. 1D RoPE over raster ids scores those two patches like horizontal neighbours. Axial 2D RoPE (from C3)
   scores every pair by (dy, dx) only.
3. Resizing a learned table for a new resolution: the cls row must be split off, and antialiasing
   changes the result when shrinking.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import rope, vision2d  # noqa: E402
from posenc.c03 import Rope2D, sincos_2d_embedding  # noqa: E402
from posenc.report import Report  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    rep = Report("04_vision_2d")
    H = W = 24
    d = 64
    rows, cols = vision2d.grid_coords(H, W)

    rep.print(f"\n1-2. A {H}x{W} patch grid. One random query vector and one key vector at every patch.")
    q, k = torch.randn(d), torch.randn(d)
    f = rope.inv_freq(d)
    rope2d = Rope2D(d, max_side=64)

    def score_1d(a, b):
        ia, ib = vision2d.raster_id(*a, W), vision2d.raster_id(*b, W)
        qa = rope.apply_half(q[None, None], torch.tensor([ia]), f)
        kb = rope.apply_half(k[None, None], torch.tensor([ib]), f)
        return float((qa * kb).sum())

    def score_2d(a, b):
        qa = rope2d(q[None, None, None], torch.tensor([a[0]]), torch.tensor([a[1]]))
        kb = rope2d(k[None, None, None], torch.tensor([b[0]]), torch.tensor([b[1]]))
        return float((qa * kb).sum())

    pairs = [
        ("right neighbour", (5, 5), (5, 6)),
        ("row wrap: end of row 5 -> start of row 6", (5, W - 1), (6, 0)),
        ("vertical neighbour", (5, 5), (6, 5)),
        ("same offsets as the wrap, elsewhere", (10, W - 1), (11, 0)),
    ]
    out = []
    for name, a, b in pairs:
        out.append([name, vision2d.raster_id(*b, W) - vision2d.raster_id(*a, W), f"({b[0] - a[0]}, {b[1] - a[1]})",
                    score_1d(a, b), score_2d(a, b)])
    rep.table(["pair", "1D id gap", "(dy, dx)", "1D RoPE score", "2D RoPE score"], out)
    rep.data["pairs"] = out

    shifts = [(0, 0), (3, 7), (10, -4)]
    base = [score_2d((r, c), (r + 2, c - 3)) for r, c in [(4 + s[0], 8 + s[1]) for s in shifts]]
    rep.print(f"2D RoPE score for offset (2, -3) at three different places: {', '.join(f'{b:.5f}' for b in base)}")

    rep.print("\n3. Resizing a position table. A smooth 24x24 table (2D sin-cos from C3) plus a cls row.")
    table = torch.cat([torch.randn(1, d), sincos_2d_embedding(H, W, d)], 0)
    up = vision2d.resize_pos_table(table, 24, 32)
    back = vision2d.resize_pos_table(up, 32, 24)
    rel = float((back[1:] - table[1:]).norm() / table[1:].norm())
    rep.print(f"   24 -> 32 -> 24 round trip: shape {tuple(up.shape)}, relative error {rel:.4f}, cls row unchanged: "
              f"{bool(torch.equal(back[0], table[0]))}")
    try:
        vision2d.resize_pos_table(table, 24, 32, has_cls=False)
    except ValueError as e:
        rep.print(f"   treating the cls row as a grid cell: ValueError: {e}")
    small_aa = vision2d.resize_pos_table(table, 24, 12, antialias=True)
    small = vision2d.resize_pos_table(table, 24, 12, antialias=False)
    gap = float((small_aa[1:] - small[1:]).norm() / small[1:].norm())
    rep.print(f"   24 -> 12 with and without antialias: relative difference {gap:.3f}")
    rep.data["resize"] = {"round_trip_rel_error": rel, "antialias_rel_difference": gap}
    rep.save()


if __name__ == "__main__":
    main()
