"""Post sections: Swin windows and the shift, Lab: reach and receptive field, TimeSformer's five schemes,
ViViT's four factorisations.

Reach is computed exactly, as boolean matrix products over the real connectivity of each block type.
Saves results/03_reach.txt and .json.
"""

import _setup  # noqa: F401
import torch

from vitfamily import reach as rc
from vitfamily.blocks import DividedSTBlock
from vitfamily.c03 import be_polite
from vitfamily.report import Report

be_polite(8)
log = Report("03_reach")
SIDE, M, DEPTH = 56, 7, 24
corner, centre = 0, (SIDE // 2) * SIDE + SIDE // 2

w0, w1 = rc.window_block(SIDE, M, 0), rc.window_block(SIDE, M, M // 2)
schemes = {
    "global": [rc.global_block(SIDE)] * DEPTH,
    "window, no shift": [w0] * DEPTH,
    "window + shift (Swin)": [w0, w1] * (DEPTH // 2),
    "shift, mask removed": [w0, rc.window_block(SIDE, M, M // 2, masked=False)] * (DEPTH // 2),
    "conv 3x3": [rc.conv_block(SIDE, 3)] * DEPTH,
    "axial (row, col)": rc.axial_blocks(SIDE) * (DEPTH // 2),
}

log(f"== Reach of one token on a {SIDE}x{SIDE} stage-1 map, M={M}: percent of the map reached after L blocks ==")
depths = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
log(f"{'scheme':<24}" + "".join(f"{f'L={d}':>7}" for d in depths) + f"{'cover from corner':>19}")
data = {}
for name, blocks in schemes.items():
    r_c = rc.reach_over_depth(blocks, centre)
    cover = rc.blocks_to_cover(blocks, corner)
    data[name] = {"centre_pct": [100 * float(r.float().mean()) for r in r_c], "blocks_to_cover_from_corner": cover}
    pct = "".join(f"{100 * float(r_c[d - 1].float().mean()):>7.1f}" for d in depths)
    log(f"{name:<24}{pct}{('never' if cover is None else str(cover)):>19}")

log("\n== Growth of the reach radius from the centre token, in tokens (Chebyshev) ==")
for name in ("window + shift (Swin)", "conv 3x3"):
    r_c = rc.reach_over_depth(schemes[name], centre)
    radii = [rc.chebyshev_radius(r, SIDE, centre) for r in r_c[:12]]
    data[name]["radius"] = radii
    log(f"{name:<24} L=1..12: {radii}")
sw = data["window + shift (Swin)"]["radius"]
cv = data["conv 3x3"]["radius"]
# Slope over L = 1..7, before the shifted-window radius saturates at the map edge (28 tokens from the centre).
log(f"radius gained per block, L=1 to L=7: shifted windows {(sw[6] - sw[0]) / 6:.2f}, conv 3x3 {(cv[6] - cv[0]) / 6:.2f}, "
    f"ratio {(sw[6] - sw[0]) / (cv[6] - cv[0]):.2f}")
wrap = rc.reach_over_depth(schemes["shift, mask removed"][:2], 0)[-1].view(SIDE, SIDE)
fixed = rc.reach_over_depth(schemes["window + shift (Swin)"][:2], 0)[-1].view(SIDE, SIDE)
log(f"corner token after W-MSA + SW-MSA reaches the far corner (55, 55)? with mask: {bool(fixed[-1, -1])}, "
    f"without mask: {bool(wrap[-1, -1])} (the torus seam)")

log("\n== One Swin window per stage: 49 tokens as a share of the map ==")
for stage, side in enumerate((56, 28, 14, 7), start=1):
    log(f"stage {stage}: {side}x{side} map, a 7x7 window covers 1/{side * side // 49} of it")

log("\n== Space-time schemes, T=8 frames of 14x14 patches (N=196), one block ==")
log(f"{'scheme':<8}{'keys per query':>16}{'pairs per head':>16}{'tokens reached':>16}")
st = {}
for scheme in ("S", "ST", "T+S", "T+W+H"):
    passes = rc.st_passes(scheme, 8, 14)
    kpq, reached = rc.keys_per_query(passes), rc.reach_after_block(passes)
    st[scheme] = {"keys_per_query": kpq, "pairs": kpq * 8 * 196, "reached": reached}
    log(f"{scheme:<8}{kpq:>16}{kpq * 8 * 196:>16,}{reached:>16} of {8 * 196}")
log(f"joint / divided pairs: {st['ST']['pairs'] / st['T+S']['pairs']:.1f}x")
log.data.update({"spatial": data, "space_time": st})

log("\n== ViViT / TimeSformer initialisation: zero temporal output projection ==")
torch.manual_seed(0)
blk = DividedSTBlock(96, 3, zero_init_temporal=True).eval()
x = torch.randn(2, 8, 49, 96)
with torch.no_grad():
    same = (blk(x) - blk.spatial_only(x)).abs().max().item()
    blk.attn_t.proj.weight.normal_(0, 0.02)
    moved = (blk(x) - blk.spatial_only(x)).abs().max().item()
log(f"max |divided block - image block per frame|: zero-init {same:.1e}, after perturbing W_O^temporal {moved:.3f}")
attn_params = sum(p.numel() for p in blk.attn.parameters())
both = attn_params + sum(p.numel() for p in blk.attn_t.parameters())
log(f"attention parameters per block: spatial only {attn_params:,}, divided {both:,} ({both / attn_params:.0f}x)")
log.save()
