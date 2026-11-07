"""Section "The KL-regularised objective": the closed-form optimum, checked on 12 completions.

1. Gradient ascent on J(p) = E_p[r] - beta KL(p || ref), with no knowledge of the answer, lands on the tilt
   ref * exp(r / beta) / Z, and the optimal value is beta log Z.
2. No random distribution beats the tilt.
3. Support: a completion with ref = 0 stays at 0, even with the largest reward in the set.
4. The beta sweep of Figure 06: high beta stays near ref, low beta piles into the best supported completion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.report import Report  # noqa: E402
from prefopt.tilt import kl, log_partition, maximise_numerically, objective, tilt  # noqa: E402


def setup():
    g = torch.Generator().manual_seed(0)
    ref = torch.rand(12, generator=g, dtype=torch.float64) ** 2
    ref[11] = 0.0  # a completion the reference model never writes
    ref = ref / ref.sum()
    r = torch.randn(12, generator=g, dtype=torch.float64)
    r[11] = 10.0  # ... and it has by far the largest reward
    return ref, r


def main():
    rep = Report("02_kl_tilt")
    ref, r = setup()
    rep.line("ref = " + " ".join(f"{x:.3f}" for x in ref.tolist()))
    rep.line("r   = " + " ".join(f"{x:+.2f}" for x in r.tolist()))
    rep.line()
    rep.line("1. Numerical maximisation (Adam on softmax logits, 4000 steps) against the closed form")
    rows, data = [], {"beta": []}
    for beta in (2.0, 0.6, 0.2):
        closed = tilt(ref, r, beta)
        numeric = maximise_numerically(ref, r, beta)
        j_closed, j_num = objective(closed, ref, r, beta).item(), objective(numeric, ref, r, beta).item()
        blz = beta * log_partition(ref, r, beta).item()
        rows.append([beta, (closed - numeric).abs().max().item(), j_closed, j_num, blz])
        data["beta"].append({"beta": beta, "max_abs_diff": rows[-1][1], "J_closed": j_closed, "J_numeric": j_num,
                             "beta_logZ": blz})
    rep.table(["beta", "max |p_closed - p_numeric|", "J(closed)", "J(numeric)", "beta log Z"], rows)

    rep.line()
    beta = 0.6
    best = objective(tilt(ref, r, beta), ref, r, beta).item()
    g = torch.Generator().manual_seed(1)
    worst_gap = float("inf")
    for _ in range(20_000):
        p = torch.softmax(torch.randn(12, generator=g, dtype=torch.float64) * 3 + ref.log(), -1)
        worst_gap = min(worst_gap, best - objective(p, ref, r, beta).item())
    rep.line(f"2. 20,000 random distributions on the support, beta = {beta}: smallest J(tilt) - J(p) = {worst_gap:.2e}")

    rep.line()
    p_star = tilt(ref, r, 0.01)
    rep.line(f"3. Support: completion 11 has ref = 0 and reward {r[11].item():.0f}. "
             f"Its probability under the tilt at beta = 0.01 is {p_star[11].item()}.")
    leak = ref.clone()
    leak[11] = 1e-9
    leak = leak / leak.sum()
    rep.line(f"   With ref = 1e-9 instead, the tilt gives it {tilt(leak, r, 0.6)[11].item():.4f} at beta = 0.6 "
             f"and {tilt(leak, r, 2.0)[11].item():.2e} at beta = 2.")

    rep.line()
    rep.line("4. The beta sweep (exact)")
    rows = []
    for beta in (10.0, 3.0, 1.0, 0.6, 0.3, 0.1, 0.03):
        p = tilt(ref, r, beta)
        ent = -(p[p > 0] * p[p > 0].log()).sum().item()
        rows.append([beta, (p * r).sum().item(), kl(p, ref).item(), ent, p.max().item(), int(p.argmax())])
    rep.table(["beta", "E_p[r]", "KL(p||ref)", "entropy", "top mass", "top item"], rows)
    best_supported = int(torch.where(ref > 0, r, -torch.inf).argmax())
    rep.line(f"   best supported completion: item {best_supported} (reward {r[best_supported].item():+.2f})")
    data["sweep"] = [dict(zip(["beta", "expected_r", "kl", "entropy", "top_mass", "top_item"], row)) for row in rows]
    data["worst_gap_random"] = worst_gap
    rep.data = data
    rep.save()


if __name__ == "__main__":
    main()
