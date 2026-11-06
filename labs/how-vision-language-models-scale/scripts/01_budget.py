"""The compute budget of one training sample, the post's Figure 1 and its insights, as arithmetic.

    python labs/how-vision-language-models-scale/scripts/01_budget.py
"""

from _common import Report, table

from vlmscale.budget import (K_FROZEN, K_FROZEN_PASS_GRAD, K_TRAINED, PRESETS, attention_flops, encoder_share,
                             llm_equivalent_of_encoder, sample_flops)

r = Report("budget")
G = 1e9

# ---------------------------------------------------------------- Figure 1
r("== Figure 1: one sample, 0.4B encoder, 448 px, 14 px patches, 2x2 merge, 200 text tokens ==")
rows = []
for llm in (7.0, 0.5):
    for name, kv, kl in (("both trained", K_TRAINED, K_TRAINED),
                         ("frozen encoder", K_FROZEN, K_TRAINED),
                         ("frozen encoder + frozen LLM (connector trains)", K_FROZEN, K_FROZEN_PASS_GRAD)):
        f = sample_flops(0.4 * G, llm * G, side=448, patch=14, merge=2, text_tokens=200, k_v=kv, k_l=kl)
        rows.append([f"{llm}B LLM, {name}", f.patches, f.visual_tokens, f"{f.encoder / 1e12:.2f}",
                     f"{f.llm / 1e12:.2f}", f"{f.total / 1e12:.2f}", f"{100 * f.encoder_share:.1f}%"])
r(table(rows, ["setting", "patches P", "tokens V", "encoder TFLOP", "LLM TFLOP", "total TFLOP", "encoder share"]))
r("Post: with a 7B LLM the encoder is about 11% of the compute; with a 0.5B LLM about 64%.")

# ---------------------------------------------------------------- closed form vs direct sum
r("\n== Encoder share of visual compute, N_V m^2 / (N_V m^2 + N_L (1 + T/V)), text ignored (T = 0) ==")
rows = []
for name in ("Qwen2-VL-2B", "Qwen2-VL-7B", "Qwen2-VL-72B", "SmolVLM-256M (r=4)", "SmolVLM-256M (r=2)",
             "LLaVA-1.5-13B", "InternVL2.5-78B"):
    p = PRESETS[name]
    closed = encoder_share(p["n_v"], p["n_l"], p["merge"])
    direct = sample_flops(p["n_v"] * G, p["n_l"] * G, side=448, patch=14 if p["merge"] < 4 else 16,
                          merge=p["merge"]).encoder_share
    rows.append([name, p["n_v"], p["n_l"], p["merge"], f"{100 * closed:.1f}%", f"{100 * direct:.1f}%",
                 f"{llm_equivalent_of_encoder(p['n_v'], p['merge']):.2f}B"])
r(table(rows, ["model", "N_V (B)", "N_L (B)", "merge m", "share (formula)", "share (summed)",
               "LLM-equivalent of encoder"]))
r("Post: Qwen2-VL-2B about 64%, Qwen2-VL-72B under 4%; a 0.675B encoder with a 2x2 merge costs what a 2.7B LLM"
  " would; SmolVLM with r = 2 still about 73%.")

# ---------------------------------------------------------------- frozen towers
r("\n== What freezing saves, FLOPs per parameter per token ==")
r(table([["trained", K_TRAINED, "forward 2 + input grad 2 + weight grad 2"],
         ["frozen, its input needs a gradient (LLM above a trained connector)", K_FROZEN_PASS_GRAD,
          "forward 2 + input grad 2"],
         ["frozen, nothing below it trains (encoder at the bottom)", K_FROZEN, "forward 2"]],
        ["tower", "k", "made of"]))
r(f"Freezing the LLM in stage 1 saves {1 - K_FROZEN_PASS_GRAD / K_TRAINED:.3f} of its compute (a third, not two"
  f" thirds). scripts/02_measure_k.py measures these k values with a FLOP counter.")

# ---------------------------------------------------------------- resolution and attention
r("\n== Resolution: both towers grow with the square of the side; attention grows with its fourth power ==")
r("Encoder: 0.67B ViT, 32 layers, width 1280, 14 px patches (Qwen2.5-VL-like). LLM 7.6B, 2x2 merge, T = 0.")
r("Windowed: 112 x 112 px windows (64 patches) in 28 layers, full attention in 4 (as in Qwen2.5-VL).")
rows = []
for side in (224, 448, 896, 1344, 1792):
    f = sample_flops(0.67 * G, 7.6 * G, side=side, patch=14, merge=2)
    full = attention_flops(f.patches, 1280, 32)
    win = attention_flops(f.patches, 1280, 32, window=64, full_layers=4)
    rows.append([side, f.patches, f.visual_tokens, f"{f.encoder / 1e12:.2f}", f"{f.llm / 1e12:.2f}",
                 f"{full / 1e12:.2f}", f"{100 * full / (full + f.encoder):.0f}%",
                 f"{win / 1e12:.2f}", f"{100 * win / (win + f.encoder):.0f}%"])
r(table(rows, ["side px", "P", "V", "enc weights TF", "LLM weights TF", "enc attn full TF", "attn share",
               "enc attn windowed TF", "attn share"]))
r("Doubling the side multiplies P, V and both weight terms by 4, and the full-attention term by 16.")

# ---------------------------------------------------------------- PaliGemma 2
r("\n== PaliGemma 2: small model at high resolution against large model at low resolution ==")
r("Configs from the SigLIP and Gemma 2 model cards: So400m (0.4B, 27 layers, width 1152); Gemma 2 2B (2.6B,")
r("26 layers, attention width 2048); Gemma 2 27B (27.2B, 46 layers, attention width 4096). 14 px patches, no merge.")
r("Assumption: 64 text tokens per example. Units: cost of the 3B model at 224 px = 1.")
siglip, llms = (0.4 * G, 27, 1152), {"3B": (2.6 * G, 26, 2048), "28B": (27.2 * G, 46, 4096)}


def pali_cost(llm: str, side: int, with_attention: bool) -> float:
    n_l, layers, d_attn = llms[llm]
    f = sample_flops(siglip[0], n_l, side=side, patch=14, merge=1, text_tokens=64)
    total = f.total
    if with_attention:
        total += attention_flops(f.patches, siglip[2], siglip[1]) + attention_flops(f.visual_tokens + 64, d_attn, layers)
    return total


rows = []
for att in (False, True):
    base = pali_cost("3B", 224, att)
    rows.append(["weights only" if not att else "weights + attention scores",
                 f"{pali_cost('3B', 896, att) / base:.1f}", f"{pali_cost('28B', 224, att) / base:.1f}",
                 f"{pali_cost('28B', 896, att) / base:.1f}"])
rows.append(["paper (measured relative cost)", "23.5", "18.9", "~156"])
r(table(rows, ["estimate", "3B @ 896 px", "28B @ 224 px", "28B @ 896 px"]))
r("The FLOP formula agrees with the paper's direction: the 3B model at 896 px costs more per example than the")
r("28B model at 224 px. It undershoots the paper's units, which are measured training cost on TPUs rather than")
r("FLOPs. We did not reproduce the paper's numbers.")
r.save()
