"""Learned queries and the three fusion patterns, for images of any length."""

import _setup  # noqa: F401
import torch

from attnzoo.config import AttentionConfig
from attnzoo.cross import GatedCrossAttentionBlock
from attnzoo.fusion import CrossAttentionFusionBlock, EarlyFusionBlock, early_fusion_sequence
from attnzoo.masks import render
from attnzoo.resampler import PerceiverResampler, QFormer

torch.manual_seed(0)
qformer = QFormer(num_queries=32, hidden_size=256, vision_size=384)
perceiver = PerceiverResampler(num_latents=32, hidden_size=256, vision_size=384)
print(f"{'image tokens Nv':>16}{'Q-Former':>18}{'Perceiver':>18}")
for nv in [196, 576, 1369]:
    img = torch.randn(1, nv, 384)
    print(f"{nv:>16}{str(tuple(qformer(img).shape)):>18}{str(tuple(perceiver(img).shape)):>18}")
print("params:", f"Q-Former {sum(p.numel() for p in qformer.parameters()):,},", f"Perceiver {sum(p.numel() for p in perceiver.parameters()):,}")

print("\nWhat the language model pays per self-attention layer, with 30 text tokens:")
for nv in [196, 576, 1369]:
    n, m = nv + 30, 32 + 30
    print(f"  Nv={nv:>4}: early fusion {n * n:>9,} scores, after a 32-token resampler {m * m:>5,}")

print("\nEarly fusion mask, 4 image tokens then 4 text tokens:")
seq, mask = early_fusion_sequence(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
print(render(mask[0, 0]))

cfg = AttentionConfig("gqa", 32, 4)
print("\nEarly fusion:    ", tuple(EarlyFusionBlock(cfg)(torch.randn(2, 4, 32), torch.randn(2, 6, 32)).shape), "(the image joins the sequence)")
print("Cross-attention: ", tuple(CrossAttentionFusionBlock(cfg)(torch.randn(2, 6, 32), torch.randn(2, 4, 32)).shape), "(text length is kept)")
gated = GatedCrossAttentionBlock(AttentionConfig("cross", 32, 4, causal=False, positional_encoding="none"))
x = torch.randn(2, 6, 32)
print(f"Gated at init:    max |output - input| = {(gated(x, torch.randn(2, 4, 32)) - x).abs().max().item():.1e} (gates closed)")
with torch.no_grad():
    gated.attn_gate.fill_(1.0)
print(f"Gated, gate open: max |output - input| = {(gated(x, torch.randn(2, 4, 32)) - x).abs().max().item():.3f}")
