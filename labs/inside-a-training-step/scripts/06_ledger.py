"""The training memory ledger, counted from the tensors of a real step.

BF16 model, FP32 master weights and Adam moments (stepanatomy.mixed.MasterWeightAdam). After one
step, every parameter carries 16 bytes of state. The activations come from 03_lifetime.py's tracker.
08_gpu_step.py measures the same ledger with the CUDA allocator at the post's layer shape.
"""

import _path  # noqa: F401

import torch

from stepanatomy import accounting as A
from stepanatomy import lifetime, models
from stepanatomy.mixed import MasterWeightAdam, ledger
from stepanatomy.report import Report

r = Report("06_ledger")
s = models.SMALL
x = models.random_batch(s)
m = models.build(s, dtype=torch.bfloat16)
opt = MasterWeightAdam(m.parameters(), lr=3e-4)
tl = lifetime.track(m, x)  # forward + backward, recording saved tensors
opt.step()

n = sum(p.numel() for p in m.parameters())
led = ledger(m, opt)
r(f"## TinyLLM (C4), L={s.layers}, H={s.hidden}, BF16 weights, {n:,} parameters, B={s.batch}, S={s.seq}")
r.table(["category", "bytes / param", "MiB"], [[k, v, v * n / 2**20] for k, v in led.items()])
r(f"post's recipe: {A.LEDGER_BYTES}, 16 bytes per parameter")
r()
per_tok = lifetime.saved_elems_per_token_per_layer(tl, s.tokens, s.hidden, 2)
r(f"activations saved at the turn: {tl.peak / 2**20:.2f} MiB, {per_tok:.2f} H BF16-sized elements per token per"
  " block. 03_lifetime.py explains why this is above 16H. In BF16 it is higher still, because C4's RMSNorm"
  " upcasts to FP32 and saves 4-byte tensors.")
r(f"model states / activations here: {led['total'] * n / tl.peak:.2f}. At the post's shape the ratio is"
  f" {A.model_state_bytes(A.LLAMA2_7B)['total'] / A.activation_bytes(A.LLAMA2_7B):.2f}: activations scale with"
  " B, S and depth, not with parameters.")
r.data.update(ledger=led, params=n, activations_bytes=tl.peak)
r.save()
