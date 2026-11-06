"""Section "What a token buys": parity answered in one token versus after a running-parity trace.

Trains a 2-layer TinyLLM from scratch for every (n, mode, seed) in the grid and writes a table.
Runs in parallel worker processes, each with a few CPU threads.

  python labs/thinking-in-tokens/scripts/01_parity_depth.py                 # the full grid in the README
  python labs/thinking-in-tokens/scripts/01_parity_depth.py --quick         # one small pair, about a minute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def _run(job: dict) -> dict:
    import torch

    torch.set_num_threads(job.pop("threads"))
    from thinktok.parity import ParityConfig, train

    return train(ParityConfig(**job), log_every=job["steps"] // 4, log=lambda s: print(s, flush=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--deep-direct", type=int, default=6, help="also train direct at n=16 with this many layers (0 = skip)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(LAB / "results" / "01_parity_depth"))
    args = ap.parse_args()
    if args.quick:
        args.ns, args.seeds, args.steps, args.deep_direct = [8], [0], 300, 0

    jobs = [
        {"n": n, "mode": mode, "layers": args.layers, "steps": args.steps, "seed": s, "threads": args.threads}
        for n in args.ns for mode in ("direct", "cot") for s in args.seeds
    ]
    if args.deep_direct:
        jobs += [{"n": 16, "mode": "direct", "layers": args.deep_direct, "steps": args.steps, "seed": s,
                  "threads": args.threads} for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, jobs))

    from thinktok.util import hardware

    hw = hardware(args.threads) + f" per run, {args.workers} runs in parallel"
    lines = [
        f"# hardware: {hw}",
        "# command: python labs/thinking-in-tokens/scripts/01_parity_depth.py" + (" --quick" if args.quick else ""),
        f"# {args.steps} AdamW steps, batch 256, fresh random strings each step; accuracy on 2000 held-out strings",
        "",
        "| n bits | format | layers | tokens written | depth bound L*K | final answer acc | first step at >=0.99 | seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["config"]["n"], r["config"]["mode"] != "direct", r["config"]["layers"], r["config"]["seed"])):
        c = r["config"]
        lines.append(f"| {c['n']} | {c['mode']} | {c['layers']} | {r['tokens_written']} | {r['serial_depth_bound']} | "
                     f"{r['final_acc']:.3f} | {r['solved_at_step'] or 'never'} | {c['seed']} |")
    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.out + ".md").write_text(text)
    Path(args.out + ".json").write_text(json.dumps({"hardware": hw, "runs": results}, indent=1) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
