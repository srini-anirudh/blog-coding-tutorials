"""Section "You must mask the world's tokens": train the same tiny model with and without the observation mask.

    python labs/the-making-of-an-ai-agent/scripts/01_masking.py               # 3 seeds x 2 objectives, CPU
    python labs/the-making-of-an-ai-agent/scripts/01_masking.py --steps 600 --seeds 1   # quick look

On a CPU the six runs go to a process pool, about 25 minutes on 8 idle cores. Pass --device cuda to run them one
after another on a GPU instead (that path is untested here: no GPU run was made).
"""

import argparse
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agentcraft.common import table, write_results
from agentcraft.masking import train


def job(args):
    masked, steps, seed, threads, device = args
    torch.set_num_threads(threads)
    return train(masked, steps, seed, eval_every=max(steps // 12, 1), device=device,
                 log=lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    jobs = [(m, a.steps, s, 2, a.device) for s in range(a.seeds) for m in (True, False)]
    if a.device == "cpu":
        with ProcessPoolExecutor(a.workers) as pool:
            runs = list(pool.map(job, jobs))
    else:
        runs = [job(j) for j in jobs]

    def final(masked, key):
        vals = [r["final"][key] for r in runs if r["masked"] == masked]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    for key in ("answer_acc", "action_nll", "world_nll"):
        (mm, ms), (um, us) = final(True, key), final(False, key)
        rows.append([key, f"{mm:.3f} +- {ms:.3f}", f"{um:.3f} +- {us:.3f}"])
    share = runs[0]["final"]["action_share"]
    curve = []
    for i, row in enumerate(runs[0]["history"]):
        acc = {m: statistics.mean(r["history"][i]["answer_acc"] for r in runs if r["masked"] == m) for m in (1, 0)}
        un = [r["history"][i] for r in runs if not r["masked"]]
        curve.append([row["step"], acc[1], acc[0],
                      statistics.mean(u["cos_unmasked_vs_policy"] for u in un),
                      statistics.mean(u["world_over_policy_norm"] for u in un)])
    text = (f"Model-written share of scored tokens in the unmasked objective: {share:.3f}\n"
            f"{a.seeds} seeds, {a.steps} steps, batch 32. Mean +- std over seeds, on 512 held-out episodes.\n\n"
            + table(rows, ["metric (held out)", "masked (policy only)", "unmasked (every token)"])
            + "\n\nAnswer accuracy during training, and how the unmasked gradient relates to the policy gradient\n"
            + "(cosine between the full unmasked gradient and its model-token part; norm of world part / model part):\n\n"
            + table(curve, ["step", "acc masked", "acc unmasked", "cos(unmasked, policy)", "|g_world| / |g_policy|"]))
    print(text)
    suffix = "" if a.device == "cpu" else "_gpu"
    write_results("01_masking" + suffix, text, {"steps": a.steps, "seeds": a.seeds, "runs": runs}, device=a.device)


if __name__ == "__main__":
    main()
