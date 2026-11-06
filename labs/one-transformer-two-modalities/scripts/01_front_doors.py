"""Sections 1 and 2: the two front doors.

Text:   a BPE tokenizer fit on a corpus, then frozen; a row lookup that is a sparse matmul; a sparse gradient.
Vision: patchify is a reshape and is exactly invertible; the projection is a Conv2d; its gradient is dense.
Then the parameter accounting, done by building the full-size models on the meta device (no memory used).

Needs wikitext-2 and the GPT-2 tokenizer from the Hugging Face Hub (small, cached after the first run).
"""

import time

import _setup
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from twomod import BlockConfig, TextTransformer, ViT
from twomod.bpe import ByteBPE
from twomod.front import PatchEmbed, count_params, onehot_lookup, patchify, unpatchify

_setup.header()
torch.manual_seed(0)
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_text = "".join(wiki["train"]["text"])
val_text = "".join(wiki["validation"]["text"])

_setup.section("1. A BPE tokenizer, fit once and frozen")
corpus = train_text[:1_000_000]
t0 = time.time()
bpe = ByteBPE.train(corpus, num_merges=1000)
print(f"trained on {len(corpus):,} characters of wikitext-2 in {time.time() - t0:.0f} s: "
      f"256 bytes + {len(bpe.merges)} merges = vocabulary {bpe.vocab_size}")
print("first 8 merges:", [bpe.vocab[256 + i].decode() for i in range(8)])
sample = val_text[:200_000]
ids = bpe.encode(sample)
print(f"validation text: {len(sample):,} characters -> {len(ids):,} tokens "
      f"({len(sample.encode()) / len(ids):.2f} bytes per token); decode(encode(x)) == x: {bpe.decode(ids) == sample}")
odd = "naïve café, 東京 ✓"
print(f"text never seen in training {odd!r}: round trip exact: {bpe.decode(bpe.encode(odd)) == odd}")
print(f"encode is deterministic: {bpe.encode(sample[:5000]) == bpe.encode(sample[:5000])}; "
      f"output dtype is int: the tokenizer has no parameters and no gradient")

gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
print("\nThe segmentation is an artifact of the corpus. Pieces per string:")
print(f"{'string':<18}{f'our BPE ({bpe.vocab_size:,} vocab)':<42}GPT-2 BPE ({gpt2.vocab_size:,} vocab)")
for s in [" the", " The", " THE", " transformers", " Transformers", " TRANSFORMERS", " patchify"]:
    ours = bpe.pieces(s)
    theirs = [gpt2.decode([i]) for i in gpt2.encode(s)]
    print(f"{s!r:<18}{len(ours):>2} {str(ours):<39}{len(theirs):>2} {theirs}")

_setup.section("2a. Token embedding: a lookup that is really a matmul")
V, d = gpt2.vocab_size, 64
E = torch.nn.Parameter(torch.randn(V, d) * 0.02)
batch_ids = torch.tensor(gpt2.encode(train_text[:400_000])[: 8 * 512]).view(8, 512)
gathered, matmul = E[batch_ids], onehot_lookup(batch_ids, E)
print(f"E[ids] vs onehot(ids) @ E, max |difference|: {(gathered - matmul).abs().max().item():.1e}")
(gathered * torch.randn_like(gathered)).sum().backward()
rows_touched = (E.grad.abs().sum(1) > 0).sum().item()
print(f"batch of 8 x 512 = {batch_ids.numel():,} GPT-2 tokens of wikitext, {batch_ids.unique().numel():,} distinct ids")
print(f"rows of E with a non-zero gradient: {rows_touched:,} of {V:,} = {rows_touched / V:.1%}")
print(f"rows that get no update this step: {V - rows_touched:,}")

_setup.section("2b. Patch embedding: a reshape, then one shared linear map (a strided Conv2d)")
images = torch.randn(2, 3, 224, 224)
patches = patchify(images, 16)
print(f"images {tuple(images.shape)} -> patches {tuple(patches.shape)}: N = (224/16)^2 = 196, P*P*C = 768")
print(f"unpatchify(patchify(x)) == x exactly: {torch.equal(unpatchify(patches, 16, 3, 224, 224), images)}")
embed = PatchEmbed(16, 3, 768)
conv = embed.as_conv()
lin_out = embed(images)
conv_out = conv(images).flatten(2).transpose(1, 2)
print(f"Linear on flattened patches vs Conv2d(3, 768, kernel=16, stride=16), max |difference|: "
      f"{(lin_out - conv_out).abs().max().item():.1e}")
(lin_out * torch.randn_like(lin_out)).sum().backward()
dense = (embed.proj.weight.grad != 0).float().mean().item()
print(f"entries of W with a non-zero gradient after one batch of 2 images: {dense:.1%}")

_setup.section("2c. Parameter accounting at full size (built on the meta device)")
with torch.device("meta"):
    llama_body = BlockConfig(dim=4096, n_heads=32, mlp_hidden=11008, mlp="swiglu", norm="rms", bias=False)
    llm = TextTransformer(100_000, llama_body, depth=32, pos="none", tie=False)
    vit_l = ViT(BlockConfig(dim=1024, n_heads=16), depth=24, image_size=224, patch=16, num_classes=1000)
total, emb = count_params(llm), llm.embed.weight.numel()
print(f"LLM: Llama-2-7B body (32 layers, d=4096, SwiGLU 11008, RMSNorm, no bias) with V=100,000, untied head")
print(f"  total {total / 1e9:.2f} B; input embedding {emb / 1e6:.1f} M = {emb / total:.1%}; "
      f"output head another {llm.head.weight.numel() / 1e6:.1f} M")
vt, pe = count_params(vit_l), count_params(vit_l.patch_embed)
print(f"ViT-L/16 at 224 (24 layers, d=1024, 1000-class head): total {vt / 1e6:.1f} M; "
      f"patch projection {pe:,} = {pe / vt:.2%}")
B, N = 8, 4096
logits_gb, hidden_gb = B * N * 100_000 * 2 / 1e9, B * N * 4096 * 2 / 1e9
print(f"bf16 activations for B={B}, N={N}: logits B x N x V = {logits_gb:.2f} GB; "
      f"one hidden state B x N x d = {hidden_gb:.2f} GB ({logits_gb / hidden_gb:.1f}x)")
