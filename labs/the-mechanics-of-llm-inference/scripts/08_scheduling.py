"""Post section 19: the eight requests of figure 15 under static and continuous batching.

A slot-accounting simulation in units of one decode step, not a timing measurement. A prefill of N_p tokens
costs max(1, N_p / 340) steps, the rule the post's figure uses, and stalls every sequence already running.

    python labs/the-mechanics-of-llm-inference/scripts/08_scheduling.py --out .../results/scheduling.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferlab.scheduling import POST_REQUESTS, continuous_batching, static_batching  # noqa: E402
from inferlab.timing import Report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="8,4,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = Report(args.out)
    rep("requests (arrival step, prompt tokens, output tokens): " + ", ".join(map(str, POST_REQUESTS)))
    rep(f"\n{'scheduler':<20} {'total steps':>11} {'slot utilisation':>16} {'mean latency':>12} "
        f"{'stalled seq-steps':>17}")
    for slots in map(int, args.slots.split(",")):
        for fn in (static_batching, continuous_batching):
            r = fn(POST_REQUESTS, slots)
            rep(f"{r.name:<20} {r.steps:>11.0f} {100 * r.utilisation:>15.1f}% {r.mean_latency(POST_REQUESTS):>12.0f} "
                f"{r.stall:>17.0f}")
    rep("\nslot utilisation = tokens produced / (slots x decode steps). latency is completion minus arrival.")
    rep("stalled seq-steps = running sequences x steps they waited on someone else's prefill; chunked prefill")
    rep("exists to spread exactly this cost. With 8 slots and 8 requests nothing is waiting, so continuous")
    rep("batching cannot raise utilisation; it still returns short requests early instead of at step 500.")
    rep.save()


if __name__ == "__main__":
    main()
