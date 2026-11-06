"""An arena in miniature: simulate pairwise votes, fit Bradley-Terry ratings, recover the truth.

Five models with known Elo ratings. Each vote picks two models at random and the winner is drawn
with P(i beats j) = 1 / (1 + 10^((R_j - R_i)/400)). We fit ratings by maximum likelihood and
compare with the truth at several vote counts. Then we add a style effect: voters also prefer the
longer answer with some strength, and one model writes long answers. The fitted ratings absorb
the style preference, which is the post's warning that an arena measures what users find
persuasive as well as what is correct.

    python labs/measuring-what-a-vlm-can-do/scripts/04_arena.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchlab.report import Report  # noqa: E402
from benchlab.stats import elo_win_prob, fit_bradley_terry  # noqa: E402

TRUE = torch.tensor([1200.0, 1100.0, 1000.0, 950.0, 750.0])
TRUE = TRUE - TRUE.mean() + 1000
NAMES = ["m0", "m1", "m2", "m3", "m4"]
g = torch.Generator().manual_seed(0)


def votes(n: int, style_bonus: float = 0.0, verbose_model: int = 3):
    i = torch.randint(0, 5, (n,), generator=g)
    j = (i + torch.randint(1, 5, (n,), generator=g)) % 5
    logit = (TRUE[i] - TRUE[j]) * torch.log(torch.tensor(10.0)) / 400
    logit = logit + style_bonus * ((i == verbose_model).float() - (j == verbose_model).float())
    i_wins = torch.rand(n, generator=g) < torch.sigmoid(logit)
    return torch.where(i_wins, i, j), torch.where(i_wins, j, i)


r = Report("04_arena")
r("true ratings:", " ".join(f"{n} {v:.0f}" for n, v in zip(NAMES, TRUE.tolist())))
r(f"check: P(m0 beats m4) = {elo_win_prob(TRUE[0].item(), TRUE[4].item()):.3f} for a "
  f"{(TRUE[0] - TRUE[4]).item():.0f}-point gap")
r(f"\n{'votes':>7s} " + " ".join(f"{n:>6s}" for n in NAMES) + "   max |error|")
for n in [200, 1000, 5000, 50000]:
    w, lo = votes(n)
    fit = fit_bradley_terry(w, lo, 5)
    r(f"{n:7d} " + " ".join(f"{v:6.0f}" for v in fit.tolist()) + f"   {(fit - TRUE).abs().max().item():6.1f}")

r("\nStyle effect: voters add a bonus to the long-answer model m3 (natural log-odds units)")
r(f"{'bonus':>7s} " + " ".join(f"{n:>6s}" for n in NAMES) + "   m3 rank")
for bonus in [0.0, 0.5, 1.0]:
    w, lo = votes(50000, style_bonus=bonus)
    fit = fit_bradley_terry(w, lo, 5)
    rank = int((fit > fit[3]).sum()) + 1
    r(f"{bonus:7.1f} " + " ".join(f"{v:6.0f}" for v in fit.tolist()) + f"   {rank}")
r("\nThe fit cannot tell a better model from a more persuasive one: both raise the win rate.")
r.save()
