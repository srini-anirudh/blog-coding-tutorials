"""Post sections: Patchify, Anatomy of a ViT block, Inside one attention head, Position, Swin patch merging, Video.

Every check runs on random tensors in a few seconds. Saves results/01_tokens.txt.
"""

import math

import _setup  # noqa: F401
import torch
import torch.nn.functional as F

from vitfamily.attention import GPSA, MHSA, attention_weights
from vitfamily.blocks import ViTBlock
from vitfamily.c03 import be_polite
from vitfamily.patch import PatchEmbed, TubeletEmbed, merge_patches, patch_embed_matmul, pixel_unshuffle_order
from vitfamily.report import Report
from vitfamily.stats import attention_entropy, mean_attention_distance

be_polite(8)
torch.manual_seed(0)
log = Report("01_tokens")

# 1. Patchify is one strided convolution.
img = torch.randn(2, 3, 224, 224)
pe = PatchEmbed(16, 768)
conv_out, mm_out = pe(img), patch_embed_matmul(img, pe.proj)
log("== Patchify: Conv2d(3, 768, kernel=16, stride=16) vs reshape + one matmul ==")
log(f"tokens N = {conv_out.shape[1]} (224^2 / 16^2 = {224 * 224 // 256}), width D = {conv_out.shape[2]}")
log(f"max |conv - matmul| = {(conv_out - mm_out).abs().max().item():.2e}  (float32 rounding only)")
cvt = torch.nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2)(img)
log(f"CvT overlapping embed, kernel 7 stride 4: {cvt.shape[2]}x{cvt.shape[3]} tokens; each sees 49 pixels, "
    f"stride area is 16, so neighbouring tokens share pixels")

# 2. Without positions, attention cannot tell where a patch is.
blk = ViTBlock(64, 4).eval()
x = torch.randn(1, 196, 64)
perm = torch.randperm(196)
pos = torch.randn(1, 196, 64)
with torch.no_grad():
    no_pos = (blk(x[:, perm]) - blk(x)[:, perm]).abs().max().item()
    with_pos = (blk(x[:, perm] + pos) - blk(x + pos)[:, perm]).abs().max().item()
log("\n== Permutation equivariance of a ViT block (196 tokens, D=64) ==")
log(f"no positions:   max |block(shuffled) - shuffled(block)| = {no_pos:.2e}")
log(f"with positions: max |block(shuffled + pos) - shuffled(block + pos)| = {with_pos:.2f}")

# 3. Two thirds of a ViT's weights are in the MLPs.
vitb = ViTBlock(768, 12)
attn_p = sum(p.numel() for p in vitb.attn.parameters())
mlp_p = sum(p.numel() for p in vitb.mlp.parameters())
log("\n== Parameter split in one ViT-B block (D=768) ==")
log(f"attention {attn_p:,} (4D^2 = {4 * 768**2:,} + biases), MLP {mlp_p:,} (8D^2 = {8 * 768**2:,} + biases)")
log(f"MLP share of attention + MLP weights: {mlp_p / (attn_p + mlp_p):.4f}")

# 4. The sqrt(d) scale: score variance and how flat a fresh softmax is.
d, n = 64, 197
q, k = torch.randn(1, 1, n, d), torch.randn(1, 1, n, d)
raw = (q @ k.transpose(-2, -1)).var().item()
log("\n== Inside one head: why divide by sqrt(d) (d=64, 197 keys, q, k ~ N(0, 1)) ==")
log(f"score variance without scale {raw:.1f}, with scale {raw / d:.3f}")
for scaled in (False, True):
    a = attention_weights(q, k, scale=scaled)
    log(f"  scaled={scaled!s:5}: row entropy {attention_entropy(a):.2f} nats (uniform = log 197 = {math.log(n):.2f}), "
        f"mean max weight {a.max(-1).values.mean():.3f}")

# 5. A fresh ViT head looks everywhere; a fresh ConViT head is a convolution.
side, patch_px = 14, 16
x = torch.randn(4, side * side, 144)  # 9 heads of width 16
with torch.no_grad():
    _, a_vit = MHSA(144, 9)(x, return_attn=True)
    gpsa = GPSA(144, 9, side)
    _, a_gpsa = gpsa(x, return_attn=True)
    gpsa.gate.fill_(10.0)  # sigmoid(10) ~ 1: positional term only
    _, a_pos = gpsa(x, return_attn=True)
uniform = torch.full((1, 1, side * side, side * side), 1 / side**2)
log("\n== Mean attention distance at initialisation, 14x14 patches of 16 px (224 px image) ==")
log(f"uniform attention (reference):     {mean_attention_distance(uniform, side, patch_px).item():.1f} px")
log(f"fresh MHSA, 9 heads:               {mean_attention_distance(a_vit, side, patch_px).mean():.1f} px (mean over heads)")
md = mean_attention_distance(a_gpsa, side, patch_px)
log(f"fresh GPSA, gate sigmoid(1)=0.73:  {md.mean():.1f} px (heads {md.min():.1f} to {md.max():.1f})")
md = mean_attention_distance(a_pos, side, patch_px)
log(f"GPSA positional term only:         {md.mean():.1f} px (heads {md.min():.1f} to {md.max():.1f})")
peak = a_pos[0, 4, 7 * side + 7].view(side, side)
log(f"  centre head, query (7,7): weight on itself {peak[7, 7]:.3f}, on the 3x3 neighbourhood {peak[6:9, 6:9].sum():.3f}")

# 6. Swin patch merging = PixelUnshuffle + 1x1 conv, up to a fixed channel order.
x = torch.randn(2, 8, 8, 96)
merged = merge_patches(x)  # (2, 4, 4, 384)
unshuf = F.pixel_unshuffle(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
diff = (merged[..., pixel_unshuffle_order(96)] - unshuf).abs().max().item()
log("\n== Swin patch merging vs PixelUnshuffle ==")
log(f"(2, 8, 8, 96) -> {tuple(merged.shape)}; max |merge[perm] - pixel_unshuffle| = {diff:.1e}: same numbers, reordered")
log("so Linear(4C, 2C) after merging is a 1x1 conv after PixelUnshuffle with its input channels permuted")
rel_table = (2 * 7 - 1) ** 2
log(f"Swin relative position bias table, M=7: (2M-1)^2 = {rel_table} entries per head")

# 7. Video tokens.
clip = torch.zeros(1, 3, 16, 224, 224)
log("\n== Video: 16 frames at 224^2, patch 16 ==")
for t in (1, 2):
    tokens = TubeletEmbed(t, 16, 8)(clip).shape[1]
    log(f"tubelet depth t={t}: {tokens} tokens, joint attention scores per head per layer {tokens**2:,}")
log.save()
