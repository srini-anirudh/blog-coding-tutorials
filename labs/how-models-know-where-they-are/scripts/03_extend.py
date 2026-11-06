"""Sections 04, 06 and 07: what happens past the training length, and what PI, NTK and YaRN change.

Part A is arithmetic on the rotation speeds, at this lab's toy scale and at a Llama-like scale.
Part B trains tiny 2-layer models on an induction-copy task at length 64 with five positional schemes,
then evaluates them at 64, 128 and 256 tokens. The RoPE model is also run with PI, NTK-aware scaling
and YaRN, with no fine-tuning. This is a toy: the papers' results are at billions of parameters.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import extend, tasks, tinylm  # noqa: E402
from posenc.report import RESULTS, Report, threads_from_env  # noqa: E402


def part_a(rep: Report, head_dim: int, base: float, train_len: int, factor: float) -> None:
    eval_len = int(train_len * factor)
    rep.print(f"\nhead_dim {head_dim}, base {base:g}, trained to {train_len}, run at {eval_len} (s = {factor:g})")
    theta = extend.scaled_inv_freq("none", head_dim, base, 1, train_len)[0]
    rows = []
    for method in ("none", "pi", "ntk", "yarn"):
        f, scale = extend.scaled_inv_freq(method, head_dim, base, factor, train_len)
        rows.append([method, float(f[0]), float(f[-1] / theta[-1]),
                     extend.out_of_range_planes(f, theta, train_len, eval_len), scale])
    rep.table(["method", "fastest plane, rad/token", "slowest plane speed / original",
               f"planes with unseen angles by {eval_len}", "logit scale"], rows)


def summary(rep: Report, runs: list[dict], eval_lens: list[int]) -> None:
    fmt = lambda a: "n/a" if a is None else round(a, 3)  # noqa: E731
    rep.table(["seed", "model", "train loss", *[f"acc @ {n}" for n in eval_lens]],
              [[r["seed"], r["model"], round(r["final_loss"], 3), *[fmt(r["accuracy"][str(n)]) for n in eval_lens]]
               for r in runs])
    if len({r["seed"] for r in runs}) > 1:
        rep.print("\nMean over seeds")
        rows = []
        for lab in dict.fromkeys(r["model"] for r in runs):
            cells = []
            for n in eval_lens:
                vals = [r["accuracy"][str(n)] for r in runs if r["model"] == lab and r["accuracy"][str(n)] is not None]
                cells.append(sum(vals) / len(vals) if vals else None)
            rows.append([lab, *map(fmt, cells)])
        rep.table(["model", *[f"acc @ {n}" for n in eval_lens]], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--train-len", type=int, default=64)
    ap.add_argument("--eval-lens", default="64,128,256")
    ap.add_argument("--pe", default="rope,alibi,none,sinusoidal,learned")
    ap.add_argument("--name", default="03_extend")
    ap.add_argument("--merge", default="", help="comma-separated result names to combine instead of training")
    args = ap.parse_args()
    threads_from_env()
    rep = Report(args.name)
    eval_lens = [int(x) for x in args.eval_lens.split(",")]

    rep.print("\nPart A. Rotation speeds under each extension method")
    part_a(rep, 32, 10_000, args.train_len, 4)
    part_a(rep, 128, 10_000, 4096, 4)
    part_a(rep, 128, 500_000, 8192, 16)

    rep.print("\nPart B. Induction copy, vocab 128, copied segment of 12 tokens, d=128, 4 heads, 2 layers")
    rep.print(f"Trained at length {args.train_len} for {args.steps} steps, batch 64, lr 3e-3. Accuracy on copied tokens.")
    runs = []
    if args.merge:
        for name in args.merge.split(","):
            part = json.loads((RESULTS / f"{name}.json").read_text())
            rep.print(f"  from results/{name}.txt: {part['command']}")
            runs += part["runs"]
        runs.sort(key=lambda r: (r["seed"], r["model"]))
        summary(rep, runs, eval_lens)
        rep.data["runs"] = runs
        rep.save()
        return
    for seed in [int(x) for x in args.seeds.split(",")]:
        for pe in args.pe.split(","):
            torch.manual_seed(seed)
            model = tinylm.TinyLM(128, pe=pe, max_len=args.train_len)
            t0 = time.time()
            losses = tinylm.train(model, tasks.make("induction", length=args.train_len), args.steps, lr=3e-3, seed=seed)
            minutes = (time.time() - t0) / 60
            for method in [""] + (["pi", "ntk", "yarn"] if pe == "rope" else []):
                accs = {}
                for n in eval_lens:
                    if pe == "learned" and n > args.train_len:
                        accs[str(n)] = None  # the table has no row for these positions
                        continue
                    fwd = {}
                    if method:
                        freqs, scale = extend.scaled_inv_freq(method, model.head_dim, 10_000, n / args.train_len,
                                                              args.train_len)
                        fwd = {"rope_freqs": freqs, "logit_scale": scale}
                    accs[str(n)] = tinylm.accuracy(model, tasks.make("induction", length=n), batches=8, **fwd)
                label = pe if not method else f"rope + {method}"
                runs.append({"seed": seed, "model": label, "final_loss": sum(losses[-50:]) / len(losses[-50:]),
                             "train_minutes": round(minutes, 2), "accuracy": accs})
                rep.print(f"  seed {seed} {label:12s} {minutes:5.1f} min  " + "  ".join(
                    f"{n}: {'n/a' if a is None else f'{a:.3f}'}" for n, a in accs.items()))
    rep.print("")
    summary(rep, runs, eval_lens)
    rep.data["runs"] = runs
    rep.save()


if __name__ == "__main__":
    main()
