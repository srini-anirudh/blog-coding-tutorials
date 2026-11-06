"""Question 2, compressing: six connectors on one realistic patch grid.

A 448 x 448 image at patch 14 gives a 32 x 32 grid, 1 024 patches. We push random features of
that shape (d_vis = 1024, an InternViT-300M / CLIP-L width) through each connector into a
d_model = 2048 language model and print tokens out, compression ratio and parameter count.
Then two small facts the post states in words:
  - pixel shuffle is a rearrangement: it can be undone exactly, so nothing is lost before the MLP;
  - average pooling is not: two different 2 x 2 neighbourhoods with the same mean give the same token.
"""

import _common  # noqa: F401  (import path)
import torch

from gallery.connectors import build_connector, n_params, pixel_shuffle

rep = _common.Report("01_connectors")
torch.manual_seed(0)
H = W = 32
D_VIS, D_MODEL = 1024, 2048
x = torch.randn(1, H * W, D_VIS)

configs = [
    ("linear", {}, "LLaVA, PaliGemma"),
    ("mlp", {}, "LLaVA-1.5, VILA"),
    ("pixel_shuffle", {"s": 2}, "InternVL 1.5+, Idefics3"),
    ("avg_pool", {"k": 2}, "Gemma 3 (4 x 4 there)"),
    ("attn_pool", {"k": 2}, "Molmo"),
    ("resampler", {"n_queries": 64}, "Flamingo, IDEFICS2, MiniCPM-V"),
]
rep(f"input: {H} x {W} = {H * W} patches, d_vis {D_VIS} -> d_model {D_MODEL}")
rep(f"{'connector':<15}{'tokens out':>11}{'ratio':>8}{'params':>13}   used by")
rows = []
for name, kw, who in configs:
    c = build_connector(name, D_VIS, D_MODEL, **kw)
    with torch.no_grad():
        y = c(x, (H, W))
    assert y.shape == (1, c.out_tokens(H, W), D_MODEL)
    ratio = H * W / y.shape[1]
    rep(f"{name:<15}{y.shape[1]:>11}{ratio:>7.0f}x{n_params(c):>13,}   {who}")
    rows.append({"connector": name, "kwargs": kw, "tokens": y.shape[1], "ratio": ratio, "params": n_params(c)})

# the resampler's token count does not depend on the input size
r = build_connector("resampler", D_VIS, D_MODEL, n_queries=64)
sizes = [(16, 16), (32, 32), (48, 64)]
with torch.no_grad():
    outs = [r(torch.randn(1, h * w, D_VIS), (h, w)).shape[1] for h, w in sizes]
rep("")
rep("resampler, 64 queries, on grids " + ", ".join(f"{h}x{w}" for h, w in sizes) + f": tokens out {outs}")
p = build_connector("pixel_shuffle", D_VIS, D_MODEL)
outs_p = [p.out_tokens(h, w) for h, w in sizes]
rep("pixel shuffle, s = 2, on the same grids:" + f" tokens out {outs_p}")

# pixel shuffle is lossless: invert it
folded = pixel_shuffle(x, (H, W), 2)                                   # 1 x 256 x 4096
unfolded = folded.view(1, H // 2, W // 2, 2, 2, D_VIS).permute(0, 1, 3, 2, 4, 5).reshape(1, H * W, D_VIS)
rep("")
rep(f"pixel shuffle undone exactly: max |x - unshuffle(shuffle(x))| = {(unfolded - x).abs().max().item():.1e}")

# average pooling is lossy: swap the patches inside each 2 x 2 window
grid = x.view(1, H, W, D_VIS)
swapped = grid.clone()
swapped[:, 0::2, 0::2], swapped[:, 1::2, 1::2] = grid[:, 1::2, 1::2], grid[:, 0::2, 0::2]
swapped = swapped.reshape(1, H * W, D_VIS)
a = build_connector("avg_pool", D_VIS, D_MODEL)
with torch.no_grad():
    d_pool = (a(x, (H, W)) - a(swapped, (H, W))).abs().max().item()
    d_shuf = (p(x, (H, W)) - p(swapped, (H, W))).abs().max().item()
rep(f"swap two patches inside every 2x2 window: avg-pool output changes by {d_pool:.1e}, "
    f"pixel-shuffle output by {d_shuf:.2f}")
rep("  (average pooling cannot tell where inside the window a feature was; pixel shuffle can)")
rep.save({"connectors": rows, "resampler_tokens_by_grid": dict(zip(map(str, sizes), outs)),
          "pool_change_after_swap": d_pool, "shuffle_change_after_swap": d_shuf})
