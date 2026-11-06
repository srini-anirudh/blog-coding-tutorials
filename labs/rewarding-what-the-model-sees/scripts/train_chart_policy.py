"""GRPO on the tiny chart VLM. Three experiments, each over several seeds, from one SFT warm start per seed.

    credit     outcome reward vs step-level checks          (credit assignment, process rewards)
    trainable  LLM only vs connector + LLM vs full model    (which parameters receive the gradient)
    dynamic    plain sampling vs DAPO-style dynamic sampling (the zero-advantage problem)

Every run reports the post's four controls before and after RL: real image, blank
image, oracle perception (readings forced to the truth), and the perception probe.

The three experiments share their baseline config, so each seed trains every distinct
config once (five runs) and the three reports are written from those runs.

    python labs/rewarding-what-the-model-sees/scripts/train_chart_policy.py              # all seeds, all reports
    python .../train_chart_policy.py --seeds 1 --train-only                              # one seed, e.g. in parallel
    python .../train_chart_policy.py --seeds 0 1 2 --report-only                         # reports from runs/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl.charts import PARTS, ChartVLM, RLConfig, encoder_probe, make_batch, sft, train_rl  # noqa: E402
from visrl.report import table, write  # noqa: E402

EXPERIMENTS = {
    "credit": [RLConfig(reward="outcome"), RLConfig(reward="step")],
    "trainable": [RLConfig(trainable="llm"), RLConfig(trainable="connector+llm"), RLConfig(trainable="full")],
    "dynamic": [RLConfig(dynamic=False), RLConfig(dynamic=True), RLConfig(dynamic=False, prompts=96)],
}
RUNS = Path(__file__).resolve().parents[1] / "runs"      # per-seed run logs, git-ignored


def label(c: RLConfig) -> str:
    sampling = "dynamic" if c.dynamic else "plain" if c.prompts == 32 else f"plain, {c.prompts} prompts"
    return f"{c.reward} / {c.trainable} / {sampling}"


def mean(xs):
    return sum(xs) / len(xs)


def spread(xs: list[float], digits: int = 3) -> str:
    return f"{mean(xs):.{digits}f} [{min(xs):.{digits}f}, {max(xs):.{digits}f}]"


def run_seed(seed: int, args, only: str | None = None) -> dict:
    """Every distinct config once for this seed, all from one SFT warm start.

    The three experiments share the baseline (outcome / connector+llm / plain), so it is trained once.
    """
    configs = {label(c): c for exp in EXPERIMENTS.values() for c in exp}
    eval_batch = make_batch(1000, torch.Generator().manual_seed(12345))
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    base = ChartVLM()
    sft(base, args.sft_steps, args.sft_examples, g)
    probe_before = encoder_probe(base, torch.Generator().manual_seed(999))
    out, t0 = {}, time.time()
    for name, c in configs.items():
        if only and only not in name:
            continue
        c.steps = args.steps
        model = copy.deepcopy(base)
        logs, evals = train_rl(model, c, torch.Generator().manual_seed(1000 + seed), eval_batch)
        out[name] = {"seed": seed, "evals": evals,
                     "useful_first20": mean([l["useful"] for l in logs[:20]]),
                     "useful_last20": mean([l["useful"] for l in logs[-20:]]),
                     "lucky_first20": mean([l["lucky"] for l in logs[:20]]),
                     "lucky_last20": mean([l["lucky"] for l in logs[-20:]]),
                     "grad_norm": {p: mean([l[p] for l in logs]) for p in PARTS},
                     "rollouts": sum(l["rollouts"] for l in logs),
                     "encoder_probe": {"before": probe_before,
                                       "after": encoder_probe(model, torch.Generator().manual_seed(999))}}
        print(f"seed {seed} {name}: {evals[-1]}  ({time.time() - t0:.0f} s)", flush=True)
    return out


def report(experiment: str, per_seed: list[dict], args) -> None:
    results = {label(c): [s[label(c)] for s in per_seed] for c in EXPERIMENTS[experiment]}
    rows, extra = [], []
    for name, recs in results.items():
        before, after = [r["evals"][0] for r in recs], [r["evals"][-1] for r in recs]
        row = [name]
        for k in ("A_img", "A_blank", "A_oracle", "P"):
            row.append(f"{mean([b[k] for b in before]):.3f} -> {spread([a[k] for a in after])}")
        rows.append(row)
        ex = [name, f"{mean([r['useful_first20'] for r in recs]):.2f} -> {mean([r['useful_last20'] for r in recs]):.2f}",
              f"{mean([r['lucky_first20'] for r in recs]):.3f} -> {mean([r['lucky_last20'] for r in recs]):.3f}",
              " / ".join(f"{mean([r['grad_norm'][p] for r in recs]):.2f}" for p in PARTS),
              f"{mean([r['rollouts'] for r in recs]):.0f}"]
        if experiment == "trainable":
            ex.append(f"{mean([r['encoder_probe']['before'] for r in recs]):.3f} -> "
                      f"{mean([r['encoder_probe']['after'] for r in recs]):.3f}")
        extra.append(ex)
    head2 = ["config (reward / trainable / sampling)", "useful groups, first 20 -> last 20 steps",
             "lucky share of rewarded rollouts, first 20 -> last 20", "mean grad norm encoder / connector / llm",
             "rollouts generated"]
    if experiment == "trainable":
        head2.append("encoder linear probe, before -> after")
    text = (f"Experiment '{experiment}': SFT warm start ({args.sft_steps} steps on {args.sft_examples} examples), "
            f"then GRPO for {args.steps} steps, 32 prompts x K = 8 per step, lr 1e-3, no KL term. Seeds {args.seeds}. "
            "Greedy evaluation on 1000 held-out charts. 'before -> after' gives the seed mean before RL and "
            "mean [min, max] after.\n\n"
            + table(rows, ["config", "A_img (real image)", "A_blank (blank image)", "A_oracle (true readings given)",
                           "P (both readings right)"])
            + "\n\n" + table(extra, head2)
            + f"\n\nThe per-seed runs were trained with {", ".join(map(str, args.train_threads))} torch threads each, CPU only; this file was "
            "written from their saved logs, so the hardware line above names the machine, not the training threads."
            + f"\nwall time for all configs of one seed: "
            f"{', '.join(str(round(t)) + ' s' for t in args.wall)} (seeds {args.seeds}, one entry per seed)")
    write(f"chart_{experiment}", text, {"args": vars(args), "runs": results})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sft-steps", type=int, default=150)
    ap.add_argument("--sft-examples", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-only", action="store_true", help="train the seeds and save runs/, no report")
    ap.add_argument("--report-only", action="store_true", help="report from runs/ saved by --train-only")
    ap.add_argument("--only", default=None, help="train only configs whose label contains this text; merge into runs/")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: at most 8)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    RUNS.mkdir(exist_ok=True)
    if not args.report_only:
        for seed in args.seeds:
            t0 = time.time()
            path = RUNS / f"chart_seed{seed}.json"
            old = json.loads(path.read_text()) if args.only and path.exists() else {"wall": 0, "runs": {}}
            rec = run_seed(seed, args, args.only)
            path.write_text(json.dumps({"wall": old["wall"] + time.time() - t0, "threads": torch.get_num_threads(),
                                        "runs": {**old["runs"], **rec}}))
    if args.train_only:
        return
    saved = [json.loads((RUNS / f"chart_seed{s}.json").read_text()) for s in args.seeds]
    args.wall = [d["wall"] for d in saved]
    args.train_threads = sorted({d.get("threads") for d in saved}, key=str)
    for exp in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
        report(exp, [d["runs"] for d in saved], args)


if __name__ == "__main__":
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    main()
