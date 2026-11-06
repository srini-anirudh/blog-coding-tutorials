"""What the extra tokens cost: K trajectories of T tokens against one trajectory of K*T (post section 3).

Part 1 prints the post's bill for a real model config: weight FLOPs and KV slots are equal,
attention FLOPs and serial decode steps are about K times larger for the deep option.
Part 2 times both shapes on a real frozen model with a KV cache, generating exactly K*T tokens each way.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ttc.cost import wide_vs_deep
from ttc.lm import chat_ids, load, sample
from ttc.report import Report, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
ap.add_argument("--shapes", default="8x128,32x64", help="comma list of KxT")
ap.add_argument("--device", default="auto")
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--tag", default="")
args = ap.parse_args()
dev = pick_device(args.device)
r = Report("04_width_vs_depth" + (f"_{args.tag}" if args.tag else f"_{dev}"))

model, tok = load(args.model, dev)
cfg = model.config
n_params = sum(p.numel() for p in model.parameters())  # a tied embedding counts once, as the LM head
kv_dim = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
r(f"model {args.model}: {n_params / 1e6:.0f}M params, {cfg.num_hidden_layers} layers, "
  f"d_model {cfg.hidden_size}, kv_dim {kv_dim}")

shapes = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
r("\n== The bill (analytic, prompt ignored) ==")
r("shape        option     weight GFLOP  attn GFLOP  decode steps  KV slots  KV MB (bf16)")
for K, T in shapes:
    b = wide_vs_deep(n_params, cfg.num_hidden_layers, cfg.hidden_size, kv_dim, K, T)
    for name in ("wide", "deep"):
        x = b[name]
        label = f"{K} x {T}" if name == "wide" else f"1 x {K * T}"
        r(f"{K}x{T:<9} {label:<10} {x.weight_flops / 1e9:12.1f} {x.attention_flops / 1e9:11.2f} "
          f"{x.decode_steps:13d} {x.kv_slots:9d} {x.kv_bytes / 2**20:12.1f}")
    q = b["deep_over_wide"]
    r(f"{'':12} deep/wide  {q['weight_flops']:12.2f} {q['attention_flops']:11.2f} {q['decode_steps']:13.2f} "
      f"{q['kv_slots']:9.2f} {q['kv_bytes']:12.2f}")

r("\n== Measured wall clock, temperature-1 sampling, no early stop, KV cache on ==")
prompt = chat_ids(tok, "Write a long story about a lighthouse keeper.")
gen = torch.Generator(device=dev).manual_seed(0)


def timed(width, depth):
    best = float("inf")
    for _ in range(args.repeats):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks, _ = sample(model, [prompt] * width, eos=set(), pad=tok.pad_token_id or 0, max_new=depth,
                         temperature=1.0, generator=gen)
        if dev == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
        assert sum(len(t) for t in toks) == width * depth
    return best


timed(1, 8)  # warm-up
r("shape      wide K x T (s)   deep 1 x KT (s)   deep/wide   tokens/s wide   tokens/s deep")
rows = []
for K, T in shapes:
    tw, td = timed(K, T), timed(1, K * T)
    r(f"{K}x{T:<8} {tw:14.2f} {td:17.2f} {td / tw:11.2f} {K * T / tw:15.0f} {K * T / td:15.0f}")
    rows.append({"K": K, "T": T, "wide_s": tw, "deep_s": td})
r(f"best of {args.repeats} runs each. Same number of generated tokens both ways.")
r.save({"model": args.model, "device": dev, "timings": rows})
