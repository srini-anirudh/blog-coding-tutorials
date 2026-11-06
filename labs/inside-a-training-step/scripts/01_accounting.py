"""Recompute every back-of-envelope number in the post from the six architecture numbers.

Each row prints the post's figure next to the computed one. No model runs here.
"""

import _path  # noqa: F401

from stepanatomy import accounting as A
from stepanatomy.report import Report

s = A.LLAMA2_7B
r = Report("01_accounting")
GB, MB = 1e9, 1e6
N = A.param_count(s)

r("## The model: L=32, H=4096, 32 heads, I=11008, V=32000, S=4096, B=1")
r(f"parameters 32(4H^2 + 3HI) + 2VH = {N:,} = {N / 1e9:.2f}B   (post: 6.74e9)")
r()
r("## One block, op by op (BF16, causal FlashAttention). Ridge point of an H100:",
  f"{A.H100_RIDGE:.0f} FLOP/byte")
ops = A.block_ops(s)
tot = sum(o.flops for o in ops)
r.table(["op", "GFLOP", "HBM MB", "FLOP/byte", "bound", "share of FLOPs"],
        [[o.name, o.flops / 1e9, o.hbm_bytes / MB, o.intensity, o.bound, f"{100 * o.flops / tot:.3f}%"] for o in ops])
cheap = sum(o.flops for o in ops if o.name not in A.PROJECTIONS and "attention" not in o.name)
silu = next(o for o in ops if o.name == "silu * mul")
r(f"projections: {100 * A.projection_share(s):.1f}% of block FLOPs (post: 92%);"
  f" without the causal saving {100 * A.projection_share(s, causal=False):.1f}%")
r(f"norms, adds, silu*mul: {100 * cheap / tot:.3f}% of FLOPs (post: under 0.05%)")
r(f"silu*mul traffic: {silu.hbm_bytes / MB:.0f} MB per layer, {32 * silu.hbm_bytes / GB:.2f} GB per step"
  " (post: 270 MB, 8.6 GB)")
r()
r("## Tensor sizes (BF16)")
r.table(["tensor", "shape", "elements (M)", "MB"],
        [[k, str(list(v[0])), v[1] / 1e6, v[2] / MB] for k, v in A.tensor_sizes(s).items()])
lg = A.tensor_sizes(s)["logits"][1]
r(f"logits in FP32, plus softmax, plus its gradient: 3 x {lg * 4 / GB:.3f} = {3 * lg * 4 / GB:.2f} GB (post: about 1.5 GB)")
r()
r("## Where 2ND stops being true")
pl = A.params_per_layer(s)
r(f"P_layer = {pl / 1e6:.1f}M, S* = P_layer / 2H = {A.attention_crossover_seq(s):,.0f} tokens (post: about 24,700)")
for seq in (4096, 32768):
    ss = A.Shape(seq=seq)
    att, proj = A.attention_flops_per_layer(ss), A.projection_flops_per_layer(ss)
    r(f"S={seq}: attention / projections = {att / proj:.3f} without masking, {att / 2 / proj:.3f} causal;"
      f" attention share of block FLOPs = {att / (att + proj):.3f} / {att / 2 / (att / 2 + proj):.3f}")
r("post: about 17% at S=4096. That matches attention / projections without masking (0.166).")
r("post: at 32k, 2ND under-counts by more than a third. The causal attention share there is 0.40.")
r()
r("## Training FLOPs and the time wall")
r(f"6ND per 4096-token step = {A.train_flops(s):.3e} FLOPs; 8ND with full recompute = {A.train_flops(s, recompute=True):.3e}")
t_fb = A.train_flops(s) / A.H100_BF16_FLOPS
r(f"forward+backward at 100% of 989 TFLOP/s: {t_fb * 1e3:.0f} ms; at 50%: {2 * t_fb * 1e3:.0f} ms (post: about 330 ms)")
run = A.run_seconds(N, 2e12, 0.45)
r(f"6ND for 2T tokens = {6 * N * 2e12:.2e} FLOPs (post: 8.1e22); at 45% MFU = {0.45 * A.H100_BF16_FLOPS:.2e} FLOP/s;"
  f" {run / 86400 / 365:.2f} years on one H100 (post: about six)")
r()
r("## The memory ledger")
for k, v in A.model_state_bytes(s).items():
    r(f"{k:15s} {v / GB:7.1f} GB")
r(f"post: 13.5 / 13.5 / 27.0 / 27.0 / 27.0, total 108 GB; overshoot of 80 GB = {A.model_state_bytes(s)['total'] / 80e9 - 1:.0%}"
  " (post: 35%)")
act = A.activation_bytes(s)
r(f"activations L*B*S*16H*2 = {act / GB:.1f} GB = {act / s.tokens / MB:.1f} MB per token (post: 17.2 GB, 4.2 MB)")
ck = A.checkpointed_activation_bytes(s)
r(f"with full checkpointing: (L + 16) H per token = {ck / GB:.2f} GB, a {act / ck:.1f}x cut (post: about 10x)")
r(f"compute with full checkpointing: 8ND / 6ND = {8 / 6:.2f}x (post: +33%, and HFU = 1.33 MFU)")
r()
r("## Precision")
r(f"BF16 unit roundoff 2^-8 = {A.bf16_unit_roundoff():.2e} (post: about 4e-3);"
  f" an Adam step near 1e-4 relative is {A.bf16_unit_roundoff() / 1e-4:.0f}x smaller")
r(f"H100 BF16 / FP32 dense peak: {A.H100_BF16_FLOPS / 1e12:.0f} / {A.H100_FP32_FLOPS / 1e12:.0f} TFLOP/s")
r()
r("## The optimizer step")
r(f"bytes per parameter: {A.ADAM_BYTES_PER_PARAM}, FLOPs about {A.ADAM_FLOPS_PER_PARAM}:"
  f" {A.ADAM_FLOPS_PER_PARAM / A.ADAM_BYTES_PER_PARAM:.2f} FLOP/byte,"
  f" {A.H100_RIDGE / (A.ADAM_FLOPS_PER_PARAM / A.ADAM_BYTES_PER_PARAM):.0f}x below the ridge (post: 0.43, about 700x)")
t_opt = A.optimizer_step_seconds(s)
r(f"traffic {N * A.ADAM_BYTES_PER_PARAM / GB:.0f} GB per step; at 2.5 TB/s {t_opt * 1e3:.0f} ms (post: 189 GB, 75 ms);"
  f" {t_opt / (t_opt + 0.33):.0%} of a step with 330 ms of compute (post: 20%)")
r.data.update(params=N, ops=[o.__dict__ for o in ops], ledger=A.model_state_bytes(s), activations=act)
r.save()
