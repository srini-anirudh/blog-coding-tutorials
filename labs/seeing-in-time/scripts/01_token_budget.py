"""The token arithmetic of video (post Figure 2), checked by computation, plus a measured attention cost.

N = D * f * tau. The script reproduces the post's numbers, then times one attention layer and one
feed-forward layer on this CPU as the sequence doubles, to show the N^2 against N growth.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from _common import Report
from seeingtime.budget import (attention_vs_ffn_growth, seconds_that_fit, slowfast_tokens, tokens_per_frame,
                               video_tokens)


def timed(fn, reps: int = 3) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=8192)
    args = ap.parse_args()
    rep = Report("01_token_budget")
    tau = tokens_per_frame(448, 448, 14, 2)
    rep.print(f"Tokens per frame, 448 x 448, 14-px patches, 2 x 2 merge: {tau} (the post: 256, InternVL 2.5)")
    rows = [[label, f"{video_tokens(d, 2, tau):,}"] for label, d in
            (("one image", 0.5), ("1 minute", 60), ("10 minutes", 600), ("1 hour", 3600))]
    rep.print("\nVisual tokens at 2 fps and 256 tokens per frame (post Figure 2):")
    rep.table(["video", "tokens"], rows)
    rep.data["figure2"] = rows

    rep.print("\nGemini arithmetic quoted in the post (1 fps):")
    rows = [["1 hour at 258 tokens/frame", f"{video_tokens(3600, 1, 258):,}"],
            ["3 hours at 66 tokens/frame", f"{video_tokens(10800, 1, 66):,}"],
            ["seconds in 1,000,000 tokens at 258/frame", f"{seconds_that_fit(1_000_000, 1, 258):,.0f}"],
            ["seconds in 1,000,000 tokens at 66/frame", f"{seconds_that_fit(1_000_000, 1, 66):,.0f}"]]
    rep.table(["quantity", "value"], rows)
    rep.data["gemini"] = rows

    sf = slowfast_tokens(10, 12 * 24, 50, 4 * 4)
    fig6 = slowfast_tokens(4, 192, 16, 16)
    rep.print(f"\nSlowFast-LLaVA: 10 x (12 x 24) + 50 x (4 x 4) = {sf:,} tokens (the post: 3,680)")
    rep.print(f"Figure 6 split: 4 slow x 192 + 16 fast x 16 = {fig6:,}; equal: 16 x 64 = {16 * 64:,}")
    rep.data["slowfast"] = {"slowfast_llava": sf, "figure6": fig6}

    ffn_x, attn_x = attention_vs_ffn_growth(2)
    rep.print(f"\nDoubling the sequence: feed-forward x{ffn_x:.0f}, attention scores x{attn_x:.0f}.")
    rep.print("Measured on this CPU: one head-dim-64, 8-head attention vs one 512 -> 2048 -> 512 MLP, batch 1.")
    torch.manual_seed(0)
    d, heads = 512, 8
    w1, w2 = torch.randn(d, 4 * d) / d ** 0.5, torch.randn(4 * d, d) / (4 * d) ** 0.5
    rows, prev = [], None
    n = 1024
    while n <= args.max_n:
        q = torch.randn(1, heads, n, d // heads)
        x = torch.randn(n, d)
        ta = timed(lambda: F.scaled_dot_product_attention(q, q, q))
        tf = timed(lambda: F.gelu(x @ w1) @ w2)
        ratio = "" if prev is None else f"x{ta / prev[0]:.1f} / x{tf / prev[1]:.1f}"
        rows.append([f"{n:,}", f"{ta * 1e3:.1f}", f"{tf * 1e3:.1f}", ratio])
        prev = (ta, tf)
        n *= 2
    rep.table(["tokens", "attention ms", "MLP ms", "growth vs previous row (attn / MLP)"], rows)
    rep.data["timing"] = rows
    rep.print("Timings are medians of 3 on a shared machine; read the growth, not the absolute values.")
    rep.save()


if __name__ == "__main__":
    main()
