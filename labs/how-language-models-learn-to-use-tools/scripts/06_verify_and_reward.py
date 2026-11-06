"""Generate, execute, verify, keep, and the same checks as an executable reward.

Each trained tiny model (from 03_tool_diversity.py) samples 8 calls at temperature 1 for each request
over unseen tools. The samples go through APIGen's three gates in order of cost. Then the same samples
are scored with r = r_fmt + r_exec + r_match, and the script counts how many groups of 8 would give a
group-relative (GRPO-style) update any signal at all: a group whose 8 rewards are equal has zero
advantage everywhere. C12 (c12-agent-rl) trains with such advantages; this script only measures them.

    python labs/how-language-models-learn-to-use-tools/scripts/06_verify_and_reward.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.evaluate import eval_set  # noqa: E402
from toollab.model import decode, load  # noqa: E402
from toollab.verify import gates, reward  # noqa: E402
from toollab.world import Vocab, make_tools, split_names  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", default=str(LAB / "results" / "verify_and_reward.json"))
    args = ap.parse_args()
    hd = header()
    ckpts = sorted((LAB / "runs").glob("tools_N*.pt"), key=lambda p: int(p.stem.split("N")[1]))
    if not ckpts:
        sys.exit("run scripts/03_tool_diversity.py first")
    vocab = Vocab()
    _, heldout = split_names(seed=0)
    exs = eval_set(make_tools(300, heldout, random.Random(7)), args.n_prompts, seed=404)
    gen = torch.Generator().manual_seed(0)
    res = {**hd, "runs": {}}
    print(f"{args.n_prompts} requests over unseen tools x {args.group} samples at temperature 1\n")
    print(f"{'N':>6} | {'fail format':>11} {'fail exec':>9} {'fail semantic':>13} {'kept':>6} | "
          f"{'mean r':>6} {'groups with signal':>18}")
    for ck in ckpts:
        model, meta = load(ck)
        funnel, rewards, signal = Counter(), [], 0
        for ex in exs:
            group = []
            for _ in range(args.group):
                out = decode(model, vocab, ex, temperature=1.0, gen=gen)
                funnel[gates(ex, out)] += 1
                group.append(reward(ex, out)["r"])
            rewards += group
            signal += statistics.pstdev(group) > 0
        total = sum(funnel.values())
        n = meta["n_tools"]
        res["runs"][n] = {"funnel": dict(funnel), "samples": total, "mean_reward": statistics.mean(rewards),
                          "groups_with_nonzero_advantage": signal, "groups": len(exs),
                          "reward_histogram": dict(Counter(rewards))}
        print(f"{n:>6} | {funnel['format'] / total:>11.3f} {funnel['execution'] / total:>9.3f} "
              f"{funnel['semantic'] / total:>13.3f} {funnel['kept'] / total:>6.3f} | "
              f"{statistics.mean(rewards):>6.3f} {signal:>11}/{len(exs)}")
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
