"""The pi0 action expert: shared attention, a block-wise causal mask, and a prefix cache reused for ten flow steps.

Checks that the cache is exact, that the prefix does not depend on the actions, and times ten flow steps with and
without the cache. Weights are random: this is about the computation, not about a trained policy.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.expert import ActionExpertPolicy, block_causal_mask  # noqa: E402
from vlalab.report import Report  # noqa: E402


def median_ms(fn, reps: int):
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=816, help="image + text tokens (3 x 256 image tokens + 48 text)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "05_action_expert"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/05_action_expert.py"] + sys.argv[1:]))
    torch.manual_seed(0)
    H, A, S = 50, 7, 8
    m = ActionExpertPolicy(A, S, H, layers=args.layers, d_vlm=512, d_exp=256, heads=8, head_dim=64,
                           mlp_vlm=2048, mlp_exp=1024).eval()
    split = m.param_split()
    rep.line(f"Model: {args.layers} layers. Backbone width 512, expert width 256, 8 heads of 64 shared. "
             f"Parameters: backbone {split['backbone'] / 1e6:.2f} M, expert {split['expert'] / 1e6:.2f} M.")
    rep.line(f"Sequence: {args.prefix} prefix tokens, 1 state token, {H} action tokens of {A} dims.")
    mask = block_causal_mask([3, 1, 2]).int()
    rep.line("Block-wise causal mask for blocks of 3 (image+text), 1 (state), 2 (actions); 1 = may attend:")
    for row in mask.tolist():
        rep.line("  " + " ".join(map(str, row)))

    prefix = torch.randn(1, args.prefix, 512)
    state = torch.randn(1, S)
    a_tau, tau = torch.randn(1, H, A), torch.tensor([0.3])
    with torch.no_grad():
        cache, px = m.encode_prefix(prefix)
        v_cached = m.velocity(cache, state, a_tau, tau)
        v_full, px_full = m.full_forward(prefix, state, a_tau, tau)
        _, px_other = m.full_forward(prefix, state, torch.randn(1, H, A), torch.tensor([0.9]))
    diff_v = (v_cached - v_full).abs().max().item()
    diff_p = (px_full - px_other).abs().max().item()
    rep.line(f"\nMax |velocity, cached - full pass|: {diff_v:.2e}")
    rep.line(f"Max change in prefix outputs when the noisy actions and tau change: {diff_p:.2e}")

    def full_ten():
        with torch.no_grad():
            a = torch.randn(1, H, A)
            for i in range(10):
                v, _ = m.full_forward(prefix, state, a, torch.tensor([i / 10]))
                a = a + v / 10

    def cached_ten():
        m.sample(prefix, state, steps=10)

    def prefix_only():
        with torch.no_grad():
            m.encode_prefix(prefix)

    t_full, t_cached, t_prefix = (median_ms(f, args.reps) for f in (full_ten, cached_ten, prefix_only))
    rows = [{"what": "10 flow steps, full pass each step", "median ms": round(t_full, 1)},
            {"what": "prefix once + 10 expert steps", "median ms": round(t_cached, 1)},
            {"what": "  of which the prefix", "median ms": round(t_prefix, 1)},
            {"what": "  of which the 10 expert steps", "median ms": round(t_cached - t_prefix, 1)}]
    rep.line(f"\nTiming, batch 1, median of {args.reps}:")
    rep.table(rows)
    rep.line(f"Speed-up from the cache: {t_full / t_cached:.1f}x. For comparison, pi0 reports on an RTX 4090: "
             "14 ms image encoding, 32 ms observation pass, 27 ms for ten expert steps (73 ms). Not reproduced here.")
    rep.save_json({"params": split, "max_diff_velocity": diff_v, "max_diff_prefix": diff_p, "timing_ms": rows,
                   "speedup": t_full / t_cached, "prefix_tokens": args.prefix, "layers": args.layers})


if __name__ == "__main__":
    main()
