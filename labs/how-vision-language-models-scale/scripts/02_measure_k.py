"""Measure k (FLOPs per parameter per token) per tower with a FLOP counter, under four freezing settings.

    python labs/how-vision-language-models-scale/scripts/02_measure_k.py

The model is a real two-tower VLM (vlmscale.model), small enough for a CPU. The counter records every
matmul in forward and backward, so this checks the post's 6 / 4 / 2 rule rather than assuming it.
"""

import torch
from _common import Report, save_json, table

from vlmscale.budget import encoder_share
from vlmscale.flops import SETTINGS, count_step, measured_encoder_share
from vlmscale.model import TwoTowerConfig, TwoTowerVLM

torch.manual_seed(0)
r = Report("measure_k")
cfg = TwoTowerConfig(image_size=64, patch_size=8, v_dim=192, v_layers=4, v_heads=4, merge=2,
                     l_dim=256, l_layers=4, l_heads=4, vocab=512, max_text=32)
model = TwoTowerVLM(cfg)
b, t = 4, 32
pixels, text = torch.randn(b, 3, 64, 64), torch.randint(0, cfg.vocab, (b, t))
r(f"Model: encoder {cfg.v_layers} layers x {cfg.v_dim}, {cfg.patches} patches; {cfg.merge}x{cfg.merge} merge ->"
  f" {cfg.visual_tokens} visual tokens; LLM {cfg.l_layers} layers x {cfg.l_dim}; {t} text tokens; batch {b}.")

rows, payload = [], {}
for name, flags in SETTINGS.items():
    c = count_step(model, pixels, text, text, *flags)
    payload[name] = {k: {"k": v.k, "weight_flops": v.weight, "attention_flops": v.attention, "params": v.params,
                         "tokens": v.tokens} for k, v in c.items()}
    rows.append([name, f"{c['vision'].k:.2f}", f"{c['connector'].k:.2f}", f"{c['llm'].k:.2f}",
                 f"{c['vision'].attention / 1e6:.1f}", f"{c['llm'].attention / 1e6:.1f}"])
r("\nk = weight-matmul FLOPs / (matmul parameters x tokens), forward + backward:")
r(table(rows, ["setting", "k encoder", "k connector", "k LLM", "enc attn MFLOP", "LLM attn MFLOP"]))
r("""
Readings:
- A trained tower costs 6. The encoder shows slightly under 6: its first layer (the patch embedding) never
  needs a gradient for the pixels.
- A frozen encoder costs 2: nothing below it trains, so no backward pass runs through it.
- A frozen LLM above a trained connector costs 4, not 2: the input gradient still flows down through it.
- The connector falls between 4 and 6 when the encoder below it is frozen: its first layer skips the input
  gradient, its second does not.
- Attention-score FLOPs in the LLM do not change when it is frozen: they have no weights to skip.""")

both = count_step(model, pixels, text, text, True, True, True)
formula = encoder_share(both["vision"].params, both["llm"].params, cfg.merge, t / cfg.visual_tokens)
r("\nEncoder share, both towers trained:")
r(table([["post's formula N_V m^2 / (N_V m^2 + N_L (1 + T/V))", f"{100 * formula:.1f}%"],
         ["measured, weight matmuls only", f"{100 * measured_encoder_share(both):.1f}%"],
         ["measured, with attention scores", f"{100 * measured_encoder_share(both, True):.1f}%"]],
        ["estimate", "encoder share"]))
r("The formula and the measurement differ only by the patch-embedding layer's skipped input gradient.")
save_json("measure_k", {"config": cfg.__dict__, "settings": payload})
r.save()
