"""The activation mountain: saved tensors pile up through the forward and drain through the backward.

Counts only tensors autograd saves for backward (parameters excluded), by hooking every save and
every release. Runs on CPU in FP32; the shape of the curve does not depend on the device.
"""

import _path  # noqa: F401
from dataclasses import replace

from stepanatomy import lifetime, models
from stepanatomy.report import Report

r = Report("03_lifetime")
s = models.SMALL
x = models.random_batch(s)
tl = lifetime.track(models.build(s, "sdpa"), x)

r(f"## TinyLLM (C4), L={s.layers}, H={s.hidden}, B={s.batch}, S={s.seq}, FP32, SDPA attention")
r(f"{len(tl.events)} save/release events. Peak {tl.peak / 2**20:.2f} MiB at event {tl.peak_index};"
  f" the loss exists at event {tl.turn}. Live after backward: {tl.events[-1].live_bytes} bytes")
r()
r(lifetime.ascii_mountain(tl))
r()
r("## Nested lifetimes: each block's saved tensors live from its forward to its backward")
rows = [[tag, a, b, b - a, tl.saved_by_tag[tag] / 2**20] for tag, (a, b) in tl.spans.items()]
r.table(["owner", "first save", "last release", "lifetime (events)", "saved MiB"], rows)
r()
r("## What one block saves (block 1)")
r.table(["shape", "dtype", "KiB"], [[str(list(sh)), dt, b / 1024] for sh, dt, b in tl.shapes_by_tag["block 1"]])
per_tok = lifetime.saved_elems_per_token_per_layer(tl, s.tokens, s.hidden, 4)
inter = s.intermediate / s.hidden
r(f"saved per token per block: {per_tok:.2f} H elements. The post counts about 16H: 8 H-wide tensors and 3 I-wide"
  f" ones, 8 + 3 x {inter:.2f} = {8 + 3 * inter:.2f}H.")
r(f"The gap is two unfused ops. Each RMSNorm also saves its value before the gain (2 x 1H), and the SiLU-multiply"
  f" saves silu(gate) as a fourth I-wide tensor ({inter:.2f}H): {8 + 3 * inter:.2f} + 2 + {inter:.2f} ="
  f" {10 + 4 * inter:.2f}H. Selective recompute in 05_checkpointing.py removes exactly these.")
r.data["mountain"] = [(e.phase, e.tag, e.live_bytes) for e in tl.events]
r.data["per_token_per_block_H"] = per_tok

r()
r("## Saved bytes per block against sequence length: naive attention keeps the S x S scores, SDPA does not")
rows = []
for seq in (128, 256, 512, 1024):
    ss = replace(s, seq=seq, batch=1)
    xx = models.random_batch(ss)
    per = {}
    for attn in ("naive", "sdpa"):
        t = lifetime.track(models.build(ss, attn), xx)
        per[attn] = t.saved_by_tag["block 2"] / 2**20
    rows.append([seq, per["naive"], per["sdpa"], per["naive"] / per["sdpa"]])
r.table(["S", "naive MiB", "sdpa MiB", "naive / sdpa"], rows)
r("SDPA grows linearly with S. Naive attention adds terms in S^2 (scores and softmax, per head).")
r.save()
