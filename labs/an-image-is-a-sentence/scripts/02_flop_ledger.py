"""Post sections: The quadratic wall, Deriving Swin's complexity, PVT, MViT, TimeSformer, Lab: the complexity bench,
Training cost vs inference cost.

Each block is built for real and run on the `meta` device under torch's FlopCounterMode, so FLOPs are measured
from the ops the module issues, with no arithmetic done. The post's formula sits next to each measurement.
Saves results/02_flop_ledger.txt and .json.
"""

import _setup  # noqa: F401
import torch

from vitfamily import flops as fl
from vitfamily.attention import MHSA, PoolingAttention, SRAttention
from vitfamily.blocks import DividedSTBlock, PVTBlock, SwinBlock, ViTBlock
from vitfamily.c03 import WindowAttention
from vitfamily.report import Report

log = Report("02_flop_ledger")
M = 1e6


def meta(build, *shape):
    with torch.device("meta"):
        mod, x = build(), torch.empty(*shape)
    return mod, x


def run(mod, *args):
    return fl.measure_flops(lambda: mod(*args))


# 1. The ViT block, measured, against 24 N D^2 + 4 N^2 D.
log("== ViT block FLOPs, D=768, measured vs 24ND^2 + 4N^2D ==")
log(f"{'image':>6} {'N':>5} {'linear meas.':>14} {'formula':>14} {'attention meas.':>16} {'formula':>14} {'attn share':>10}")
rows = []
for px in (224, 384, 512, 1024, 1088):
    n = (px // 16) ** 2 + 1
    got = run(*meta(lambda: ViTBlock(768, 12), 1, n, 768))
    ref = fl.vit_block_flops(n, 768)
    share = got["attention"] / got["total"]
    rows.append({"px": px, "N": n, **{f"meas_{k}": v for k, v in got.items()}, "formula": ref})
    log(f"{px:>6} {n:>5} {got['linear']:>14,} {ref['linear']:>14,} {got['attention']:>16,} {ref['attention']:>14,} {share:>10.3f}")
log(f"crossover N = 6D = {fl.crossover_tokens(768)} tokens = {(fl.crossover_tokens(768)) ** 0.5:.1f}^2 patches "
    f"= {16 * fl.crossover_tokens(768) ** 0.5:.0f} px at patch 16")
log.data["vit_block"] = rows

# 2. Swin's table at stage 1 (h = w = 56, D = 96, M = 7) and stage 4 (h = w = 7, D = 768), in MACs.
log("\n== Swin attention sublayer, MACs (FLOPs / 2), measured vs Swin eq. (1) and (2) ==")
swin = {}
for name, side, d, heads in (("stage 1", 56, 96, 3), ("stage 4", 7, 768, 24)):
    n = side * side
    glob = run(*meta(lambda: MHSA(d, heads), 1, n, d))
    win = run(*meta(lambda: WindowAttention(d, heads, 7, 0), 1, side, side, d))
    sw = run(*meta(lambda: WindowAttention(d, heads, 7, 3 if side > 7 else 0), 1, side, side, d))
    f_g, f_w = fl.swin_msa_macs(side, side, d), fl.swin_wmsa_macs(side, side, d, 7)
    swin[name] = {"global": glob, "window": win, "shifted": sw, "formula_global": f_g, "formula_window": f_w}
    log(f"{name}: h=w={side}, D={d}")
    log(f"  projections 4hwD^2:   global {glob['linear'] / 2 / M:8.1f} M   window {win['linear'] / 2 / M:8.1f} M   (formula {f_g['linear'] / M:.1f} M)")
    log(f"  attention:            global {glob['attention'] / 2 / M:8.1f} M   window {win['attention'] / 2 / M:8.1f} M   "
        f"(formulas {f_g['attention'] / M:.1f} M and {f_w['attention'] / M:.1f} M)")
    log(f"  sublayer total:       global {glob['total'] / 2 / M:8.1f} M   window {win['total'] / 2 / M:8.1f} M   "
        f"ratio {glob['total'] / win['total']:.1f}x; shifted window {sw['total'] / 2 / M:.1f} M (the shift adds no MACs)")
    log(f"  score memory / head:  global {n * n / M:.2f} M   window {n * 49 / M:.2f} M")
s1 = swin["stage 1"]
blk_g = run(*meta(lambda: ViTBlock(96, 3), 1, 3136, 96))
blk_w = run(*meta(lambda: SwinBlock(96, 3, 7, 0), 1, 56, 56, 96))
log(f"stage 1 full block, MLP included: global {blk_g['total'] / 2 / M:.1f} M MACs, window {blk_w['total'] / 2 / M:.1f} M MACs, "
    f"ratio {blk_g['total'] / blk_w['total']:.1f}x (the post's 13.8x leaves the MLP out)")
s4 = swin["stage 4"]
log(f"stage 4: attention is {s4['global']['attention'] / s4['global']['total']:.3f} of the attention sublayer "
    f"({s4['global']['attention'] / 2 / M:.1f} M vs {s4['global']['linear'] / 2 / M:.1f} M MACs)")
log.data["swin"] = swin

# 3. PVT spatial-reduction attention at stage 1 (56 x 56, D = 64).
log("\n== PVT stage 1, 56x56 tokens, D=64 ==")
plain = run(*meta(lambda: MHSA(64, 1), 1, 3136, 64))
log(f"global attention term: {plain['attention'] / 1e9:.2f} GFLOPs per layer (post: 4 x 3136^2 x 64 = {4 * 3136**2 * 64 / 1e9:.2f} G)")
pvt = {}
for r in (8, 4, 2, 1):
    got = run(*meta(lambda: PVTBlock(64, 1, r), 1, 3136, 64), (56, 56))
    ours, post = fl.sra_block_flops(3136, 64, r), fl.sra_block_flops_post(3136, 64, r)
    pvt[r] = {"measured": got, "op_by_op": ours, "post_row": post}
    log(f"R={r}: attention {got['attention'] / M:7.1f} M (post 4N^2D/R^2 {post['attention'] / M:7.1f} M) | "
        f"rest {(got['linear'] + got.get('conv', 0)) / M:6.1f} M (op by op {(ours['linear'] + ours['conv']) / M:6.1f} M, "
        f"post 24ND^2 + ND^2/R^2 {post['rest'] / M:6.1f} M)")
with torch.no_grad():
    _, a = SRAttention(64, 1, 8)(torch.randn(1, 3136, 64), (56, 56), return_attn=True)
log(f"SRA score matrix at R=8: {tuple(a.shape[-2:])}, N x N/R^2, not square")
log.data["pvt"] = pvt

# 4. MViT pooling attention: one operator, two jobs.
log("\n== MViT pooling attention, 56x56 tokens, D=96 ==")
x = torch.randn(1, 3136, 96)
for sq, skv in ((1, 1), (1, 4), (2, 1), (2, 4)):
    with torch.no_grad():
        out, hw = PoolingAttention(96, 1, sq, skv)(x, (56, 56))
    got = run(*meta(lambda: PoolingAttention(96, 1, sq, skv), 1, 3136, 96), (56, 56))
    log(f"stride_q={sq} stride_kv={skv}: output {out.shape[1]:>4} tokens ({hw[0]}x{hw[1]}), "
        f"keys {(56 // skv) ** 2 if skv > 1 else 3136:>4}, attention {got['attention'] / M:7.1f} M FLOPs, pooling convs {got.get('conv', 0) / M:.2f} M")

# 5. Video: joint vs divided space-time attention, T = 8, N = 196, D = 768.
log("\n== Space-time attention, N=196 patches per frame, D=768 ==")
video = {}
for t in (8, 32):
    pairs = fl.st_pairs(t, 196)
    joint = run(*meta(lambda: ViTBlock(768, 12), 1, t * 196, 768))
    div = run(*meta(lambda: DividedSTBlock(768, 12), 1, t, 196, 768))
    video[t] = {"pairs": pairs, "joint": joint, "divided": div}
    log(f"T={t}: pairs per head joint {pairs['joint'] / M:.2f} M, divided {pairs['divided'] / M:.2f} M, "
        f"ratio {pairs['joint'] / pairs['divided']:.1f}x")
    log(f"      measured attention FLOPs ratio {joint['attention'] / div['attention']:.1f}x; "
        f"projection + MLP: joint {joint['linear'] / (t * 196 * 768**2):.0f} TND^2, divided {div['linear'] / (t * 196 * 768**2):.0f} TND^2 "
        f"(post's table: 24 and 48)")
log.data["video"] = video

# 6. Memory: the score matrix and the training activations.
log("\n== Score-matrix and activation memory, in elements per sample ==")
for px in (224, 1024):
    n = (px // 16) ** 2 + 1
    e = fl.score_elems_global(n, 12)
    log(f"ViT-B, {px}^2, patch 16: 12 x {n}^2 = {e:,} scores per layer = {2 * e / 2**20:.0f} MiB in fp16 ({2 * e / 1e6:.0f} MB)")
for px in (224, 1024):
    n = (px // 16) ** 2 + 1
    act = fl.vit_train_activation_elems(n, 1024, 24)
    scores = fl.score_elems_global(n, 16) * 24
    log(f"ViT-L, {px}^2: 13ND x L = {act / M:,.1f} M ({2 * act / 1e6:,.0f} MB fp16), naive scores hN^2 x L = {scores / M:,.1f} M, "
        f"scores / everything else = {scores / act:.2f}")
log.save()
