"""Backward costs twice the forward, measured: one GEMM in, two out.

FlopCounterMode counts the matmul FLOPs autograd really issues. It does not count elementwise ops,
which is fine here: they are a rounding error in FLOPs (see 01_accounting.py).
"""

import _path  # noqa: F401
from dataclasses import replace

from stepanatomy import flops as F
from stepanatomy import models, recompute
from stepanatomy.report import Report

r = Report("02_flops")

r("## Y = XW, X [512, 256], W [256, 384]. Forward is 2*S*H*H' =", 2 * 512 * 256 * 384)
rows = []
for name, ig, wg in [("both grads", True, True), ("frozen W (dW skipped)", True, False),
                     ("first layer (dX skipped)", False, True)]:
    f = F.linear_flops(512, 256, 384, input_grad=ig, weight_grad=wg)
    rows.append([name, f["forward"], f["backward"], f["backward"] / f["forward"]])
r.table(["case", "forward FLOPs", "backward FLOPs", "backward / forward"], rows)

s = models.SMALL
x = models.random_batch(s)
r()
r(f"## TinyLLM (C4), L={s.layers}, H={s.hidden}, I={s.intermediate}, V={s.vocab}, B={s.batch}, S={s.seq}."
  " Naive attention, so QK^T and PV are plain matmuls")
m = models.build(s, "naive")
N, D = F.matmul_params(m), s.tokens
f = F.model_flops(m, x)
attn_fwd = 4 * s.batch * s.seq**2 * s.hidden * s.layers  # the counter counts the full S x S, mask or not
r(f"matmul parameters N = {N:,} (embedding lookup excluded), tokens D = {D}")
r(f"forward  {f['forward']:.4e} = {f['forward'] / (N * D):.3f} ND; predicted 2ND + 4BS^2HL = "
  f"{(2 * N * D + attn_fwd) / (N * D):.3f} ND")
r(f"backward {f['backward']:.4e} = {f['backward'] / f['forward']:.3f} x forward")
r(f"total    {f['total'] / (N * D):.3f} ND (6ND plus the attention term)")
r.data["tinyllm"] = {"N": N, "D": D, **f}

r()
r("## The attention term grows with S: forward / 2ND, measured and predicted 1 + 2LSH/N")
rows = []
for seq in (64, 256, 1024, 2048):
    ss = replace(s, seq=seq, batch=1)
    mm = models.build(ss, "naive")
    ff = F.model_flops(mm, models.random_batch(ss))
    n = F.matmul_params(mm)
    rows.append([seq, ff["forward"] / (2 * n * seq), 1 + 2 * ss.layers * seq * ss.hidden / n])
r.table(["S", "measured", "predicted"], rows)
p_layer = 4 * s.hidden**2 + 3 * s.hidden * s.intermediate
r(f"S* = P_layer / 2H = {p_layer / (2 * s.hidden):.0f} for this shape: past it, attention FLOPs exceed a block's GEMMs")

r()
r("## Freezing. Backward FLOPs relative to the fully trainable model")
base = f["backward"]
rows = []
for name, frozen in [("freeze block 1 only (a frozen middle layer)", ["layers.1."]),
                     ("freeze embeddings + blocks 0-1 (a frozen bottom tower)", ["embed_tokens", "layers.0.", "layers.1."])]:
    mm = models.build(s, "naive")
    for pname, p in mm.named_parameters():
        if any(pname.startswith(k) for k in frozen):
            p.requires_grad_(False)
    ff = F.model_flops(mm, x)
    rows.append([name, ff["backward"] / base])
r.table(["case", "backward / trainable backward"], rows)
r("A frozen middle block still passes dX down, so only its dW GEMMs vanish. A frozen bottom block needs"
  " neither, so its whole backward vanishes.")

r()
r("## Recomputation. Total training FLOPs as a multiple of N*D")
rows = []
for mode in ("none", "selective", "full"):
    mm = recompute.apply(models.build(s, "naive"), mode)
    ff = F.model_flops(mm, x)
    rows.append([mode, ff["total"] / (N * D), ff["total"] / f["total"]])
r.table(["checkpointing", "FLOPs / ND", "vs none"], rows)
r("Full recompute reruns each block's forward, but not the LM head, so it lands under 8/6 = 1.33x. It also skips"
  " each block's last GEMM (down_proj): torch's checkpoint stops recomputing once every saved tensor is rebuilt,"
  " and nothing saves that GEMM's output. tests/test_mechanisms.py checks this FLOP for FLOP.")
r(
  "Selective recompute reruns only elementwise ops, which the counter does not count.")
r.save()
