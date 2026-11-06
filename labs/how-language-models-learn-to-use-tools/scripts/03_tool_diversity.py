"""Lookup or interpretation? Train the same tiny model on N distinct tools and test it on tools it never saw.

Every run sees the same number of training examples M (steps x batch). Only N, the number of distinct
tools those examples are drawn from, changes. The post predicts that when M/N is large the model
memorises its tools and fails on new ones, and that when M/N is small only reading the description works.

    python labs/how-language-models-learn-to-use-tools/scripts/03_tool_diversity.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.evaluate import eval_set, score, summarise  # noqa: E402
from toollab.model import build_model, decode, save, train  # noqa: E402
from toollab.world import Vocab, make_tools, split_names  # noqa: E402


def evaluate(model, vocab, seen_pool, unseen_pool, n_eval, train_names, max_tools=3) -> dict:
    out = {}
    for split, pool, seed in (("seen", seen_pool, 101), ("unseen", unseen_pool, 202)):
        exs = eval_set(pool, n_eval, seed, max_tools=max_tools)
        rows = [score(ex, decode(model, vocab, ex), train_names) for ex in exs]
        out[split] = summarise(rows)
        # Selection difficulty grows with the number of tools in the context.
        by_k = {}
        for k in range(1, max_tools + 1):
            sub = [r for ex, r in zip(exs, rows) if len(ex.tools) == k and ex.gold is not None]
            by_k[k] = round(sum(r["exact"] for r in sub) / max(1, len(sub)), 4)
        out[split]["exact_on_calls_by_tools_in_context"] = by_k
    return out


def table(paths: list[Path]) -> None:
    runs = []
    for p in paths:
        d = json.loads(p.read_text())
        runs += [(d["hardware"], d["command"], r) for r in d["runs"].values()]
    runs.sort(key=lambda x: x[2]["n_tools"])
    for hw, cmd, r in runs:
        print(f"N = {r['n_tools']}: {hw}\n  {cmd}")
    print(f"\n{'N':>6} {'M/N':>8} | exact: {'seen':>6} {'unseen':>7} | unseen: {'decide':>6} {'select':>6} {'args':>6} "
          f"| unseen wrong names: {'not in context':>14} {'a training tool':>15}")
    for _, _, r in runs:
        u, f = r["unseen"], (lambda v: "-" if v is None else f"{v:.3f}")
        print(f"{r['n_tools']:>6} {r['examples_per_tool']:>8} |        {r['seen']['exact']:>6.3f} {u['exact']:>7.3f} | "
              f"        {f(u['decide']):>6} {f(u['select_given_call']):>6} {f(u['args_given_tool']):>6} "
              f"|                     {f(u['wrong_name_not_in_context']):>14} {f(u['wrong_name_is_training_tool']):>15}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tools", type=int, nargs="+", default=[16, 128, 1024, 4000])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-tools", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 keeps the default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="default: results/tool_diversity_N<n>.json, one file per N")
    ap.add_argument("--table", nargs="*", default=None, help="print the summary table from these JSON files")
    ap.add_argument("--ckpt-dir", default=str(LAB / "runs"))
    args = ap.parse_args()
    if args.table is not None:
        table([Path(p) for p in args.table] or sorted((LAB / "results").glob("tool_diversity_N*.json")))
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    vocab = Vocab()
    train_names, heldout_names = split_names(seed=0)
    unseen_pool = make_tools(300, heldout_names, random.Random(7))
    hd = header()
    for n in args.n_tools:
        cfg = dict(vars(args))
        ck = Path(cfg["ckpt_dir"]).resolve()
        cfg["ckpt_dir"] = str(ck.relative_to(LAB)) if ck.is_relative_to(LAB) else ck.name  # no absolute paths
        results = {**hd, "config": cfg, "runs": {}}
        pool = make_tools(n, train_names, random.Random(args.seed + n))
        model = build_model(len(vocab), hidden=args.hidden, layers=args.layers, heads=args.heads).to(args.device)
        t0 = time.time()
        print(f"== N = {n} tools, M = {args.steps * args.batch} examples, M/N = {args.steps * args.batch / n:.1f}")
        hist = train(model, pool, vocab, steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed,
                     device=args.device, max_tools=args.max_tools)
        secs = time.time() - t0
        names = {t.name_str for t in pool}
        ev = evaluate(model, vocab, pool, unseen_pool, args.n_eval, names, args.max_tools)
        print(f"   seen exact {ev['seen']['exact']:.3f}   unseen exact {ev['unseen']['exact']:.3f}   "
              f"({secs:.0f}s training)", flush=True)
        results["runs"][str(n)] = {"n_tools": n, "examples_per_tool": round(args.steps * args.batch / n, 1),
                                   "train_seconds": round(secs, 1), "loss": hist, **ev}
        save(model, Path(args.ckpt_dir) / f"tools_N{n}.pt", {"n_tools": n, "seed": args.seed})
        out = Path(args.out or LAB / "results" / f"tool_diversity_N{n}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
