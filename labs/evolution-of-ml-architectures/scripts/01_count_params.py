"""How big is a model, really: parameters, bytes and FLOPs from a config, checked against real modules.

Every closed-form count is compared with `sum(p.numel())` of a module built in PyTorch or Hugging Face
`transformers` (on the meta device, so nothing is allocated and nothing is downloaded).
"""

import _setup  # noqa: F401

import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, MambaConfig, ResNetConfig
from transformers import ResNetForImageClassification
from transformers.models.mamba.modeling_mamba import MambaMixer

from archlab import counts as C
from archlab.env import LAB, start

start("01_count_params")

d = 1024
print(f"1. One layer of each family at width d = {d}")
formulas, modules = C.family_formulas(d), C.family_modules(d)
with torch.device("meta"):
    hf_mamba = MambaMixer(MambaConfig(hidden_size=d, state_size=16, expand=2, conv_kernel=4,
                                      time_step_rank=64, use_bias=False, use_conv_bias=True), layer_idx=0)
modules["Mamba (E=2)"] = hf_mamba
post_d2 = {"MLP": 1, "Conv 3x3": 9, "Depthwise sep. 3x3": 1, "Elman RNN": 2, "GRU": 6, "LSTM": 8,
           "Transformer": 12, "Mamba (E=2)": 6}
print(f"   {'family':>20} {'formula':>11} {'torch module':>13} {'post: k d^2':>12} {'k d^2':>11}")
for name, f in formulas.items():
    m = C.n_params(modules[name])
    print(f"   {name:>20} {f:>11,} {m:>13,} {post_d2[name]:>9} d^2 {post_d2[name] * d * d:>11,}")
print("   nn.RNN, nn.GRU and nn.LSTM keep two bias vectors per gate (b_ih and b_hh), so they hold d more per gate.")
print("   The Mamba row is the Hugging Face MambaMixer; in/out projections are 6 d^2, the rest is"
      f" {100 * (formulas['Mamba (E=2)'] - 6 * d * d) / (6 * d * d):.1f}% on top.")
print()

print("2. Llama-3-8B from six numbers")
cfg = C.LLAMA3_8B
ours = C.decoder_params(**cfg)
with torch.device("meta"):
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=cfg["V"], hidden_size=cfg["d"], intermediate_size=cfg["d_ff"],
                                         num_hidden_layers=cfg["L"], num_attention_heads=cfg["n_h"],
                                         num_key_value_heads=cfg["n_kv"], tie_word_embeddings=False))
for k, v in ours.items():
    print(f"   {k:>16}: {v:>15,}")
print(f"   {'HF LlamaForCausalLM':>16}: {C.n_params(llama):>15,}")
sys.path.insert(0, str(LAB.parent.parent / "c04-llm"))  # C4 builds this model; reuse its counter as a third check
from tinyllm.config import ModelConfig  # noqa: E402
from tinyllm.params import count_parameters  # noqa: E402

c4 = count_parameters(ModelConfig.load(LAB.parent.parent / "c04-llm" / "configs" / "llama3-8b.json"))["total"]
print(f"   {'C4 count_parameters':>16}: {c4:>15,}")
c70 = C.decoder_params(**C.LLAMA3_70B)["total"]
print(f"   Llama-3-70B by the same formula: {c70 / 1e9:.2f} B")
print()

print("3. Embeddings as a share of the model")
with torch.device("meta"):
    gpt2 = GPT2LMHeadModel(GPT2Config())
wte = gpt2.transformer.wte.weight.numel()
print(f"   GPT-2 small: {C.n_params(gpt2) / 1e6:.1f} M parameters (head tied), token embedding {wte / 1e6:.1f} M"
      f" = {100 * wte / C.n_params(gpt2):.0f}%")
print(f"   Llama-3-70B: embedding + head = {100 * C.decoder_params(**C.LLAMA3_70B)['embeddings'] / c70:.1f}%,"
      f" input embedding alone = {100 * 128256 * 8192 / c70:.2f}%")
for v0, v1 in [(32_000, 256_000), (32_768, 262_144)]:
    extra = (v1 - v0) * 4096
    print(f"   vocabulary {v0:,} -> {v1:,} at d = 4096: +{extra / 1e9:.2f} B tied, +{2 * extra / 1e9:.2f} B untied")
print()

print("4. The CNN history in parameters")
for name, model in [("LeNet-5", C.lenet5()), ("AlexNet", C.alexnet()), ("VGG-16", C.vgg16())]:
    conv, dense = C.split_conv_dense(model)
    print(f"   {name:>8}: total {C.n_params(model):>11,}   conv {conv:>11,}   dense {dense:>11,}")
with torch.device("meta"):
    r50 = ResNetForImageClassification(ResNetConfig(num_labels=1000))
print(f"   {'ResNet-50':>8}: total {C.n_params(r50):>11,}   (Hugging Face ResNetConfig defaults, 1000 classes)")
print()

print("5. From parameters to bytes, Llama-3-8B")
n8 = ours["total"]
print(f"   bf16 weights: {2 * n8 / 1e9:.1f} GB   training state at 16 B/param: {C.training_bytes(n8) / 1e9:.1f} GB")
kv_tok = C.kv_cache_bytes(cfg["L"], 1, cfg["n_kv"], cfg["d_h"])
print(f"   KV cache per token: {kv_tok:,} B = {kv_tok / 1024:.0f} KiB")
for n_ctx in [8192, 131072]:
    print(f"   KV cache at {n_ctx:>6} tokens: {C.kv_cache_bytes(cfg['L'], n_ctx, cfg['n_kv'], cfg['d_h']) / 2**30:.1f} GiB")
print(f"   tokens (batch x context) at which the cache equals the bf16 weights: {2 * n8 / kv_tok:,.0f}")
for n_kv, label in [(64, "MHA"), (8, "GQA")]:
    b = C.kv_cache_bytes(80, 128_000, n_kv, 128)
    print(f"   Llama-3-70B, {label} (n_kv = {n_kv:>2}), 128k = 128,000 tokens: {b / 1e9:.1f} GB")
print()

print("6. Attention FLOPs against parameter FLOPs, per token per layer: N_ctx / (6 d)")
for n_ctx in [2048, 4096, 24576, 131072]:
    r = C.attention_to_param_flops(n_ctx, 4096)
    ffn = (4 * n_ctx * 4096) / (2 * 8 * 4096 * 4096)
    print(f"   d = 4096, N = {n_ctx:>6}: attention / all matmuls = {r:.3f}   attention / FFN alone = {ffn:.3f}")
