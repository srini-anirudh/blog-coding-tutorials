"""AI feedback in a loop (post Figure 8): sample, let an imperfect judge pick, run DPO, repeat.

    python labs/aligning-vision-language-models/scripts/judge_loop.py --rounds 6 --seeds 0 1

The judge sees the exact truth but reports each error only with some probability, by type, and it likes detail:
    score = - (detected errors) + detail * (number of claims)
We run two detail bonuses, 0.5 and 1.0.
Each round: sample 4 responses per scene from the current policy, take the best and worst by judge score as
(y_w, y_l), and run DPO with the current policy as the reference. We then count TRUE errors per response with greedy
decoding on held-out scenes. Two judges: one that rarely catches count errors (0.15) and one that checks counts
claim by claim (0.9). Everything else is the same.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from groundpref.metrics import score_response  # noqa: E402
from groundpref.model import sample  # noqa: E402
from groundpref.pairs import Pairs  # noqa: E402
from groundpref.prefopt import VARIANTS, train_pref  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.train import describe_eval, eval_scenes, load_sft, set_threads  # noqa: E402
from groundpref.world import decode, render, sample_scene  # noqa: E402

KEYS = ["chair_i", "colour_err", "count_err", "count_per_response", "claims_per_response", "recall"]


def judge(scene, ids, detect: dict[str, float], detail: float, rng: random.Random) -> float:
    claims, _ = decode(ids)
    errs = score_response(scene, claims)
    caught = sum(sum(rng.random() < detect[k] for _ in range(errs[k])) for k in ("object", "colour", "count"))
    return -caught + detail * len(claims)


def one_round(policy, detect, detail, n_scenes, k, seed):
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    scenes = [sample_scene(rng) for _ in range(n_scenes)]
    imgs = torch.stack([render(s, gen) for s in scenes])
    samples = sample(policy, imgs.repeat_interleave(k, 0), temperature=1.0, gen=gen)
    p = Pairs([], [], [], [], [])
    for i, s in enumerate(scenes):
        cand = samples[i * k:(i + 1) * k]
        scores = [judge(s, c, detect, detail, rng) for c in cand]
        hi, lo = max(range(k), key=scores.__getitem__), min(range(k), key=scores.__getitem__)
        if scores[hi] == scores[lo] or cand[hi] == cand[lo]:
            continue
        p.scenes.append(s)
        p.imgs.append(imgs[i])
        p.yw.append(cand[hi])
        p.yl.append(cand[lo])
        p.kinds.append("judged")
    p.imgs = torch.stack(p.imgs)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--scenes", type=int, default=512)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--details", nargs="+", type=float, default=[0.5, 1.0])
    args = ap.parse_args()
    set_threads()
    log = Log("judge_loop")
    base = load_sft(log)
    scenes, imgs = eval_scenes(600, seed=3000)
    judges = {"count-blind judge": {"object": 0.9, "colour": 0.9, "count": 0.15},
              "claim-checking judge": {"object": 0.9, "colour": 0.9, "count": 0.9}}
    out = {}
    r0 = describe_eval(base, scenes, imgs)
    for detail in args.details:
        for jname, detect in judges.items():
            key = f"{jname}, detail {detail}"
            for seed in args.seeds:
                policy, rows = base, [r0]
                log(f"\n{jname} {detect}, detail bonus {detail}, seed {seed}")
                log("  round 0  " + "  ".join(f"{k}={r0[k]:.3f}" for k in KEYS))
                for rnd in range(1, args.rounds + 1):
                    pairs = one_round(policy, detect, detail, args.scenes, args.k, seed * 100 + rnd)
                    policy, _ = train_pref(policy, pairs, VARIANTS["dpo"], steps=args.steps,
                                           seed=seed * 100 + rnd, log=lambda *a: None)
                    r = describe_eval(policy, scenes, imgs)
                    r["pairs"] = len(pairs)
                    rows.append(r)
                    log(f"  round {rnd}  " + "  ".join(f"{k}={r[k]:.3f}" for k in KEYS) + f"  pairs={len(pairs)}")
                out.setdefault(key, []).append(rows)
    log("\nmean over seeds, round 0 -> last round (chair_i, colour_err, count_err are per claim)")
    for key, runs in out.items():
        parts = []
        for k in KEYS:
            a = statistics.mean(r[0][k] for r in runs)
            b = statistics.mean(r[-1][k] for r in runs)
            parts.append(f"{k} {a:.3f} -> {b:.3f}")
        log(f"  {key:34s} " + "; ".join(parts))
    log.save({"args": vars(args), "judges": judges, "results": out})


if __name__ == "__main__":
    main()
