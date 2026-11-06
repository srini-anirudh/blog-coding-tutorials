"""Is gated cross-attention slow to learn the fine question, or unable to?

In 06_fusion_freeze.py the gated cross-attention recipe is far behind on the cell-lookup question after
2 500 steps. This trains the same recipe for 2 500, 5 000 and 10 000 steps (one seed, same data stream
prefix) and reports accuracy. The gates start at zero, so early gradients into the new layers are small.
"""

import _common
import torch

from gallery.model import TinyVLM
from gallery.train import evaluate, fresh_copy, pretrain_lm, train_vlm

_common.threads()
rep = _common.Report("08_xattn_longer")
lm = pretrain_lm(seed=0, steps=1500, cache=_common.RUNS / "lm_seed0.pt")
rep(f"{'steps':>7}{'fine':>8}{'coarse':>8}{'gate_attn (tanh)':>20}{'train s':>9}")
rows = []
for steps in (2500, 5000, 10000):
    torch.manual_seed(0)
    vlm = TinyVLM(fresh_copy(lm), connector="mlp", fusion="xattn")
    info = train_vlm(vlm, ("fine", "coarse"), "connector", steps=steps, seed=0)
    gates = [torch.tanh(b.xattn.gate_attn).item() for b in vlm.lm.blocks]
    row = {"steps": steps, "fine": evaluate(vlm, "fine", True), "coarse": evaluate(vlm, "coarse", True),
           "tanh_gate_attn_per_layer": gates, **info}
    rep(f"{steps:>7}{row['fine']:>8.3f}{row['coarse']:>8.3f}{'  ' + ' '.join(f'{g:+.2f}' for g in gates):>20}{info['train_seconds']:>9.0f}")
    rows.append(row)
rep("(the learning-rate schedule is cosine over the run, so each row is its own run, not a checkpoint of the longest one)")
rep.save(rows)
