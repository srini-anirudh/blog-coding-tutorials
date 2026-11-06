"""Recompute the post's Table 2 (order-of-magnitude training FLOPs of open VLMs) from the reported inputs.

    python labs/how-vision-language-models-scale/scripts/03_compute_ladder.py
"""

from _common import Report, table

from vlmscale.ladder import CROSS_CHECKS, LADDER, PEAK_BF16, UTILISATION, order_of_magnitude_match

r = Report("compute_ladder")
r(f"Hardware method: chip-hours x peak bf16 x {UTILISATION:.0%}. Peaks (TFLOP/s): "
  + ", ".join(f"{k} {v / 1e12:.0f}" for k, v in PEAK_BF16.items()) + ". Token method: 6 N D.")
ref = LADDER[-1][3]
rows = [[m, rep, meth, f"{f:.2e}", f"{p:.0e}", "yes" if order_of_magnitude_match(f, p) else "NO",
         f"{f / ref:.1e}"] for m, rep, meth, f, p in LADDER]
r(table(rows, ["model", "paper reports", "method", "recomputed", "post", "matches", "/ Qwen2.5-72B text"]))
r("\nCross-checks named in the post:")
r(table([[k, f"{v:.2e}"] for k, v in CROSS_CHECKS.items()], ["check", "FLOPs"]))
intern, qvl = dict((m, f) for m, _, _, f, _ in LADDER)["InternVL2.5-78B"], LADDER[10][3]
r(f"\nInternVL2.5-78B multimodal training / its Qwen2.5-72B backbone's text pretraining: {intern / ref:.2%}"
  " (post: under 1%).")
r(f"Qwen2.5-VL-72B pretraining / the same backbone: {qvl / ref:.0%} (post: roughly a quarter).")
r(f"LLaVA-1.5-13B / Qwen2.5-72B text: {LADDER[1][3] / ref:.0e}, about five orders of magnitude (post: five).")
r("These are the post's own estimates recomputed; no training compute was measured here.")
r.save()
