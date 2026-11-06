"""Do personas decorrelate errors? A real small model, voted.

One model answers arithmetic questions 15 times each, in two ways:
  sampled:  the same prompt, 15 samples at temperature 0.8
  personas: 15 different persona system prompts, greedy decoding
Then we vote over N of the 15 answers and compare with what independent agents would get.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/06_persona_vote.py --device cuda
"""

import argparse
import itertools
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord.lm import DEFAULT_MODEL, Agent  # noqa: E402
from mascoord.report import Report  # noqa: E402
from mascoord.voting import implied_rho, p_majority  # noqa: E402

PERSONAS = [
    "You are a careful accountant.", "You are a high-school maths teacher.", "You are a physicist.",
    "You are a cautious engineer who double-checks everything.", "You are a fast mental-arithmetic champion.",
    "You are a patient tutor for children.", "You are a statistician.", "You are a software developer.",
    "You are a chess grandmaster.", "You are a pharmacist measuring doses.", "You are a retired banker.",
    "You are a sceptical auditor.", "You are a helpful assistant.", "You are an astronomer.",
    "You are a carpenter who measures twice.",
]
K = len(PERSONAS)
NS = [1, 3, 5, 9, 15]

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--questions", type=int, default=80)
ap.add_argument("--subsets", type=int, default=300)
args = ap.parse_args()
out = Report("06_persona_vote", args.device)
agent = Agent(args.model, device=args.device)
out(f"model: {args.model}, {args.questions} questions per task, K = {K} answers per question")

rng = random.Random(0)
TASKS = {
    "add 3-digit": [(f"What is {a} + {b}?", a + b) for a, b in
                    ((rng.randint(100, 999), rng.randint(100, 999)) for _ in range(args.questions))],
    "multiply 2-digit": [(f"What is {a} * {b}?", a * b) for a, b in
                         ((rng.randint(11, 99), rng.randint(11, 99)) for _ in range(args.questions))],
}
SUFFIX = " Reply with only the number."


def parse(reply: str):
    m = re.search(r"-?\d[\d,]*", reply)
    return int(m.group(0).replace(",", "")) if m else None


def vote_stats(answers: list[list], truth: list[int]) -> dict:
    """Plurality and strict-majority accuracy over random N-subsets of each question's K answers."""
    r = np.random.default_rng(0)
    res = {}
    for n in NS:
        plur, strict, anyc = [], [], []
        for ans, t in zip(answers, truth):
            subsets = [list(range(K))] if n == K else [r.choice(K, n, replace=False) for _ in range(args.subsets)]
            for idx in subsets:
                picked = [ans[i] for i in idx]
                c = Counter(a for a in picked if a is not None).most_common()
                top = c[0][1] if c else 0
                winners = [a for a, cnt in c if cnt == top]
                plur.append(len(winners) == 1 and winners[0] == t)
                strict.append(sum(a == t for a in picked) > n / 2)
                anyc.append(any(a == t for a in picked))
        res[n] = (np.mean(plur), np.mean(strict), np.mean(anyc))
    return res


summary = {}
for task, qs in TASKS.items():
    for mode in ["sampled", "personas"]:
        prompts = [q + SUFFIX for q, _ in qs]
        answers = [[None] * K for _ in qs]
        for j in range(K):
            if mode == "sampled":
                reps = agent.ask(prompts, system=None, max_new_tokens=12, temperature=0.8, seed=1000 + j)
            else:
                reps = agent.ask(prompts, system=PERSONAS[j], max_new_tokens=12)
            for i, rep in enumerate(reps):
                answers[i][j] = parse(rep)
        truth = [t for _, t in qs]
        correct = np.array([[a == t for a in row] for row, t in zip(answers, truth)], dtype=float)  # (Q, K)
        p = correct.mean()
        per_q = correct.mean(1)
        # mean pairwise correlation of correctness between two agents on the same question
        pairs = [np.corrcoef(correct[:, a], correct[:, b])[0, 1] for a, b in itertools.combinations(range(K), 2)
                 if correct[:, a].std() > 0 and correct[:, b].std() > 0]
        stats = vote_stats(answers, truth)
        out(f"\n{task}, {mode}: per-agent accuracy p = {p:.3f}, mean pairwise error correlation = {np.mean(pairs):.3f}")
        out(f"   questions all {K} agents got right: {(per_q == 1).mean():.2f}, all wrong: {(per_q == 0).mean():.2f}")
        out(f"{'N':>6} {'plurality':>10} {'strict maj':>11} {'indep. prediction':>18} {'best-of-N (verifier)':>21} {'indep.':>7}")
        for n in NS:
            pl, st, an = stats[n]
            out(f"{n:>6} {pl:>10.3f} {st:>11.3f} {p_majority(n, p):>18.3f} {an:>21.3f} {1 - (1 - p) ** n:>7.3f}")
        ceiling = stats[K][1]
        if p > 0.5:
            out(f"   strict majority of all {K} = {ceiling:.3f}; reading it as 1 - rho(1-p) gives rho ~ "
                f"{implied_rho(ceiling, p):.2f}")
        summary[f"{task} / {mode}"] = {"p": round(p, 4), "pair_corr": round(float(np.mean(pairs)), 4),
                                      "vote": {n: [round(x, 4) for x in stats[n]] for n in NS},
                                      "indep_majority": {n: round(p_majority(n, p), 4) for n in NS}}
out(f"\nbill: {agent.bill()}")
out.json({"model": args.model, "tasks": summary})
