"""Section 2: where the 6 in C = 6ND comes from, checked against the FLOPs PyTorch actually dispatches.

Part 1 counts one forward and one backward pass of real TinyLLM models with torch's FlopCounterMode.
Part 2 reproduces the post's Figure 3: three published FLOP conventions, and how they fan apart as width shrinks.

    python labs/scaling-laws-from-first-principles/scripts/01_six_flops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scalelab import flops  # noqa: E402
from scalelab.ladder import LADDER, build  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    print(f"# hardware: CPU, torch {torch.__version__} (FLOP counts do not depend on hardware)")
    print("# command: python scripts/01_six_flops.py\n")
    print("Part 1. Measured FLOPs per token, one forward + one backward pass, sequence 256.")
    print("P = matmul weights (blocks + tied LM head). Attention scores/values are counted separately.\n")
    print(f"{'model':9s} {'P':>10s} {'fwd/P':>7s} {'bwd/P':>7s} {'(fwd+bwd)/P':>12s} "
          f"{'attn fwd / 4LdS':>15s} {'attn bwd / fwd':>14s} {'attn share':>10s}")
    for s in (LADDER[0], LADDER[2], LADDER[5], LADDER[9]):
        g = flops.measure(build(s), batch=2, seq=s.seq, vocab=s.vocab)
        p = s.linear_params + s.vocab * s.d
        share = (g["fwd_attn"] + g["bwd_attn"]) / sum(g.values())
        print(f"{s.name:9s} {p:10,d} {g['fwd_matmul'] / p:7.3f} {g['bwd_matmul'] / p:7.3f} "
              f"{(g['fwd_matmul'] + g['bwd_matmul']) / p:12.3f} {g['fwd_attn'] / (4 * s.layers * s.d * s.seq):15.3f} "
              f"{g['bwd_attn'] / g['fwd_attn']:13.2f}x {share:10.1%}")
    print("\nThe linear part is exactly 2 FLOPs per weight forward and 4 backward: 6 per weight per token.")
    print("Attention is a separate, sequence-length cost. Its backward counts 2.5x its forward here, because")
    print("PyTorch's fused attention backward recomputes the score matrix QK^T instead of storing it.")
    print("RMSNorm, SiLU, softmax and residual adds are not matmuls and FlopCounterMode does not count them.\n")

    print("Part 2. FLOPs per token under three published conventions, relative to the reference count")
    print("(reference = 72 L d^2 + 6 V d + 6 L d S: linear layers, untied head, causal attention).")
    print("L = 32, V = 100,352, S = 8,192, d_ffn = 4d, untied embeddings: the post's Figure 3 defaults.\n")
    print(f"{'d':>6s} {'reference':>11s} {'Kaplan 6N_ne':>13s} {'Chinchilla 6N':>14s} {'DeepSeek M':>11s}  head+attn share")
    for d in (4096, 2048, 1024, 512, 256):
        c = flops.conventions(d, 32, 100_352, 8_192)
        ref = c["reference"]
        f = flops.per_token(d, 32, 100_352, 8_192)
        share = (f["head"] + f["attn_causal"]) / ref
        print(f"{d:6d} {ref:11.3e} {c['kaplan'] / ref:13.3f} {c['chinchilla'] / ref:14.3f} "
              f"{c['deepseek_M'] / ref:11.3f}  {share:14.1%}")
    print("\nAt d = 4096 the conventions agree within about 30%. As d shrinks, the head and attention, which")
    print("6N_non-embedding ignores, become most of the work, so Kaplan's count undershoots more and more.")
    f = flops.per_token(4096, 1, 1, 8192)
    print(f"Attention / linear FLOPs at d = 4096, S = 8192: unmasked 12dS/72d^2 = {f['attn_unmasked'] / f['linear']:.3f}"
          f" (the post's S/(6d)), causal 6dS/72d^2 = {f['attn_causal'] / f['linear']:.3f}.")
    print("Note: FlopCounterMode counts attention unmasked (the full S x S score matrix), as DeepSeek's M does.")


if __name__ == "__main__":
    main()
