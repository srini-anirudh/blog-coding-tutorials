"""The truncation zoo on real next-token distributions, then truncation as desmoothing on a toy where the truth is known.

    python labs/choosing-the-next-token/scripts/04_truncation_zoo.py

Part 1 runs SmolLM2-135M over WikiText-2 passages and applies all six candidate-set rules at every position.
It reports how each set's size moves with entropy, and how often the real next token survives the cut.
Part 2 trains small softmax-bottleneck models on distributions with a known, limited support (synthetic).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import desmoothing, dists, lm, truncation as tr  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402

RULES = {"top_k": 50, "top_p": 0.9, "epsilon": 3e-4, "eta": 3e-4, "min_p": 0.1, "typical": 0.9}
r = Report(__file__, "04_truncation_zoo")
tok, model = lm.load(lm.BASE)


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


r("== Part 1: six rules on 32 WikiText-2 passages, every position, SmolLM2-135M ==")
H, sizes, hits = [], {k: [] for k in RULES}, {k: [] for k in RULES}
for prompt, ref in prompt_and_reference(tok, 32, prompt_tokens=1, ref_tokens=100):
    ids = torch.cat([prompt[0], ref])[None]
    p = torch.softmax(lm.all_logits(model, ids)[0, :-1], -1)  # predict tokens 1..100
    nxt = ids[0, 1:]
    H.append(dists.entropy_bits(p))
    for rule, param in RULES.items():
        keep = tr.RULES[rule](p, param)
        sizes[rule].append(keep.sum(-1).float())
        hits[rule].append(keep.gather(-1, nxt[:, None])[:, 0].float())
H = torch.cat(H)
sizes = {k: torch.cat(v) for k, v in sizes.items()}
hits = {k: torch.cat(v) for k, v in hits.items()}
bins = [(0, 0.5), (0.5, 2), (2, 4), (4, 6), (6, 8), (8, 99)]
r(f"positions: {H.numel()}.  Median |C| per entropy bin (bits), with the count of positions per bin:")
r(f"{'rule':<16}" + "".join(f"{f'{lo}-{hi if hi < 99 else 99}':>10}" for lo, hi in bins))
r(f"{'(positions)':<16}" + "".join(f"{int(((H >= lo) & (H < hi)).sum()):>10}" for lo, hi in bins))
for rule, param in RULES.items():
    cells = [sizes[rule][(H >= lo) & (H < hi)] for lo, hi in bins]
    r(f"{rule + ' ' + str(param):<16}" + "".join(f"{float(c.median()) if c.numel() else float('nan'):>10.0f}"
                                                 for c in cells))
r(f"\n{'rule':<16} {'spearman(H, |C|)':>17} {'mean |C|':>9} {'|C| = 1':>8} {'real next token kept':>21}")
for rule, param in RULES.items():
    r(f"{rule + ' ' + str(param):<16} {spearman(H, sizes[rule]):17.2f} {sizes[rule].mean():9.1f} "
      f"{(sizes[rule] == 1).float().mean():8.3f} {hits[rule].mean():21.3f}")

r("\n== Three prompts: candidate-set sizes when one token dominates and when none does ==")
for text in ["Donald", "The capital of France is", "import numpy as"]:
    ids = tok(text, return_tensors="pt").input_ids
    p = torch.softmax(lm.next_logits(model, ids), -1)
    top = torch.topk(p[0], 4)
    r(f"{text!r}: H = {dists.entropy_bits(p)[0]:.2f} bits; top tokens "
      + ", ".join(f"{tok.decode(i)!r} {v:.3f}" for v, i in zip(top.values, top.indices)))
    r("   |C|: " + ", ".join(f"{k} {int(tr.RULES[k](p, v).sum())}" for k, v in RULES.items()))

r("\n== Order of operations: min-p 0.1 at high temperature, mean |C| over the same positions ==")
logits = torch.cat([lm.all_logits(model, torch.cat([p_[0], ref])[None])[0]
                    for p_, ref in prompt_and_reference(tok, 8, prompt_tokens=1, ref_tokens=100)])
for T in (1.0, 1.5, 3.0):
    a = tr.candidate_mask(logits, "min_p", 0.1, T, "truncate_then_temper").sum(-1).float()
    b = tr.candidate_mask(logits, "min_p", 0.1, T, "temper_then_truncate").sum(-1).float()
    r(f"T={T}: truncate-then-temper {a.mean():8.1f}   temper-then-truncate {b.mean():8.1f}")

r("\n== Part 2: truncation as desmoothing (synthetic; the true support is known) ==")
C, V, MAXS, S = 300, 2000, 300, 400
p_true = desmoothing.make_true_distributions(C, V, MAXS, seed=0)
support_size = (p_true > 0).sum(-1)
sharp, broad = support_size <= 5, support_size >= 100
toy_rules = {"top_k": 20, "top_p": 0.9, "epsilon": 1e-3, "eta": 1e-3, "min_p": 0.05, "typical": 0.9}
r(f"{C} contexts, vocabulary {V}, support sizes 1 to {MAXS} (log-spaced), {S} training samples per context.")
r(f"sharp contexts (support <= 5): {int(sharp.sum())}; broad contexts (support >= 100): {int(broad.sum())}")
for d in (8, 32, 128):
    model_d = desmoothing.train(p_true, d=d, samples_per_context=S, steps=3000)
    with torch.no_grad():
        p_model = torch.softmax(model_d(torch.arange(C)), -1)
    rep = desmoothing.support_report(p_model, p_true, toy_rules)
    r(f"\nbottleneck d={d}: mean mass outside the true support {rep['leaked_mass_mean']:.4f}; "
      f"smallest model probability anywhere {rep['min_model_prob']:.2e}")
    rs, rb = (desmoothing.support_report(p_model, p_true, toy_rules, rows) for rows in (sharp, broad))
    r(f"{'rule':<14} {'precision':>9} {'recall':>7} {'|C|':>7} | {'sharp: prec':>11} {'|C|':>6} | "
      f"{'broad: recall':>13} {'|C|':>6}")
    for k in toy_rules:
        a, s_, b = rep["rules"][k], rs["rules"][k], rb["rules"][k]
        r(f"{k + ' ' + str(toy_rules[k]):<14} {a['precision']:9.3f} {a['recall']:7.3f} {a['mean_set_size']:7.1f} | "
          f"{s_['precision']:11.3f} {s_['mean_set_size']:6.1f} | {b['recall']:13.3f} {b['mean_set_size']:6.1f}")
r.save()
