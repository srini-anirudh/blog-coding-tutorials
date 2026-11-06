"""Attention dropout, biases, QK-norm and soft-capping: inspect the logits they produce."""

import _setup  # noqa: F401
import torch

from attnzoo.attention_naive import attention_dropout
from attnzoo.biases import alibi_bias, alibi_slopes
from attnzoo.config import AttentionConfig
from attnzoo.stability import RMSNorm, logit_stats, soft_cap
from attnzoo.variants import CosineAttention, DifferentialAttention, GatedAttention, TalkingHeadsAttention

torch.manual_seed(0)
d = 64
norm = RMSNorm(d)
q0, k0 = torch.randn(1, 8, 256, d), torch.randn(1, 8, 256, d)


def row(name: str, scores: torch.Tensor) -> None:
    s = logit_stats(scores)
    print(f"  {name:<20}{s['max_logit']:>10.1f}{s['logit_std']:>10.2f}{s['mean_max_prob']:>10.3f}{s['mean_entropy']:>10.2f}")


print("Logits as activations grow. Uniform attention over 256 keys has entropy 5.55.")
for growth in [1.0, 4.0, 16.0]:
    q, k = q0 * growth, k0 * growth
    raw = (q @ k.transpose(-2, -1)) / d**0.5
    print(f"activation scale x{growth:g}{'':<14}{'max |s|':>6}{'std':>10}{'max p':>10}{'entropy':>10}")
    row("plain", raw)
    row("QK-norm", (norm(q) @ norm(k).transpose(-2, -1)) / d**0.5)
    row("soft-cap c=5", soft_cap(raw, 5.0))

print("\nDropout on one row of weights (p = 0.5): links vanish, survivors double")
w = torch.softmax(torch.randn(6), -1)
print("  before:", w, "sum", round(w.sum().item(), 3))
after = attention_dropout(w, 0.5)
print("  after: ", after, "sum", round(after.sum().item(), 3))

print("\nALiBi for 8 heads: the slopes, and the bias of head 0 and head 7 for the last of 5 queries")
slopes = alibi_slopes(8)
print("  slopes:", slopes)
bias = alibi_bias(slopes, 5, 5)
print("  head 0 (steep): ", bias[0, 0, 4])
print("  head 7 (shallow):", bias[0, 7, 4])

print("\nResearch variants, one forward pass each (hidden 64, 4 heads):")
cfg = AttentionConfig("mha", 64, 4)
x = torch.randn(1, 8, 64)
for cls in [GatedAttention, CosineAttention, DifferentialAttention, TalkingHeadsAttention]:
    m = cls(cfg)
    print(f"  {cls.__name__:<24}{str(tuple(m(x).shape)):<14}{sum(p.numel() for p in m.parameters()):>7,} params")
