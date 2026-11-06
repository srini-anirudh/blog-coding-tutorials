"""Chunk size, inference latency and reaction to a surprise, in a toy 50 Hz loop (the post's Figure 3).

Also replays pi0's reported schedule: 73 ms on-board inference, a new chunk every 0.5 s, 25 actions executed per chunk.
This is a didactic model, not a measured robot.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.loop import LoopConfig, sweep  # noqa: E402
from vlalab.report import Report  # noqa: E402


def rounded(rows):
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "04_chunk_latency_loop"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/04_chunk_latency_loop.py"] + sys.argv[1:]))
    cfg = LoopConfig()
    rep.line(f"50 Hz for {cfg.seconds} s. Object bumped by {cfg.bump} between 1.6 and 2.74 s (three steps later per "
             f"seed, {args.seeds} seeds). Each call adds an offset with std {cfg.offset_std}.")
    rep.line("calls_per_s: policy calls. paused_frac: steps with nothing to execute. reaction_ms: time from the bump "
             "to the first command computed after it. jump_at_switch / step_elsewhere: mean hand step at chunk "
             "switches and at other steps. track_err: mean |hand - object|.")
    rows = rounded(sweep([1, 5, 10, 25, 50], [20, 80, 200], ["sync", "async", "blend"], seeds=args.seeds))
    for lat in (20, 80, 200):
        rep.line(f"\nInference latency {lat} ms:")
        rep.table([{k: v for k, v in r.items() if k != "latency_ms"} for r in rows if r["latency_ms"] == lat])
    rep.line("\npi0's reported schedule (73 ms, async, H = 25 executed per 0.5 s) against waiting for each chunk:")
    pi0 = rounded(sweep([25], [73], ["sync", "async"], seeds=args.seeds))
    rep.table(pi0)
    rep.save_json({"sweep": rows, "pi0_schedule": pi0})


if __name__ == "__main__":
    main()
