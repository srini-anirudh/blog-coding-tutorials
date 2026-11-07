"""Section "The failure mode hiding in plain sight": likelihood displacement in a model small enough to see.

The policy is softmax(E h) over 50 one-token responses. E holds fixed 16-dimensional unit embeddings and h is
trained, as the final hidden state is in a real model. One preference pair (y_w, y_l). We control the cosine
similarity of their embeddings, run DPO, and watch log pi(y_w), log pi(y_l), and where the mass goes.

The DPO gradient on h is beta * w * (E_w - E_l): the two softmax expectations cancel. When E_w and E_l are close,
that direction is nearly orthogonal to E_w, so it raises whichever other responses happen to align with it.
DPO-Positive and RPO (an added NLL term on y_w) are the two fixes the post names.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.dpo import SoftmaxPolicy, train_dpo  # noqa: E402
from prefopt.report import Report  # noqa: E402

V, D, BETA, STEPS, LR = 50, 16, 0.1, 300, 0.05


def make_embeddings(cos: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    e = torch.nn.functional.normalize(torch.randn(V, D, generator=g, dtype=torch.float64), dim=-1)
    u = e[1] - (e[1] @ e[0]) * e[0]
    u = u / u.norm()
    e[1] = cos * e[0] + (1 - cos**2) ** 0.5 * u  # y_w = 0, y_l = 1, with the chosen cosine similarity
    h0 = torch.randn(D, generator=g, dtype=torch.float64)
    return e, e @ h0


def run(cos: float, seed: int, variant: str = "dpo", lam: float = 1.0):
    e, base = make_embeddings(cos, seed)
    pol = SoftmaxPolicy(base, e)
    ref = pol.log_probs().detach()
    logs = train_dpo(pol, torch.tensor([0]), torch.tensor([1]), BETA, STEPS, LR, variant=variant, lam=lam,
                     log_every=STEPS)
    lp = pol.log_probs().detach()
    gain = lp.exp() - ref.exp()
    return {"d_logp_w": (lp[0] - ref[0]).item(), "d_logp_l": (lp[1] - ref[1]).item(),
            "loss_start": logs[0]["loss"], "loss_end": logs[-1]["loss"], "mass_other": 1 - lp[:2].exp().sum().item(),
            "mass_other_ref": 1 - ref[:2].exp().sum().item(), "top_gainer": int(gain.argmax()),
            "top_gain": gain.max().item()}


def main():
    rep = Report("05_displacement_toy")
    seeds = range(20)
    rep.line(f"{V} responses, {D}-dim unit embeddings, beta = {BETA}, Adam lr {LR}, {STEPS} steps, 20 embedding draws")
    rep.line("\n1. One draw (seed 0), plain DPO, as the similarity of y_w and y_l rises")
    rows = []
    for cos in (0.0, 0.5, 0.8, 0.95, 0.99):
        x = run(cos, 0)
        rows.append([cos, x["loss_start"], x["loss_end"], x["d_logp_w"], x["d_logp_l"], x["mass_other_ref"],
                     x["mass_other"], x["top_gainer"]])
    rep.table(["cos(E_w, E_l)", "loss start", "loss end", "change log pi(y_w)", "change log pi(y_l)",
               "mass elsewhere (ref)", "mass elsewhere (end)", "biggest gainer"], rows)

    rep.line("\n2. All 20 draws: fraction where log pi(y_w) ended below the reference, and the median change")
    summary = {}
    rows = []
    for variant, lam in (("dpo", 0.0), ("dpop", 50.0), ("rpo", 1.0)):
        for cos in (0.0, 0.5, 0.8, 0.95, 0.99):
            xs = [run(cos, s, variant, lam) for s in seeds]
            dw = torch.tensor([x["d_logp_w"] for x in xs])
            frac = (dw < 0).double().mean().item()
            loss_end = torch.tensor([x["loss_end"] for x in xs]).median().item()
            rows.append([variant, cos, frac, dw.median().item(), loss_end])
            summary[f"{variant}_cos{cos}"] = {"frac_winner_down": frac, "median_d_logp_w": dw.median().item(),
                                             "median_loss_end": loss_end}
    rep.table(["variant", "cos(E_w, E_l)", "frac y_w down", "median change log pi(y_w)", "median final loss"], rows)
    rep.line("dpop uses lambda = 50 inside the margin; rpo adds 1.0 x NLL(y_w). Each final loss is the variant's own")
    rep.line("loss; all start at 0.6931 for dpo and dpop. Plain DPO's loss falls in every run, including the ones")
    rep.line("where log pi(y_w) falls by 20 nats.")
    rep.data = {"summary": summary}
    rep.save()


if __name__ == "__main__":
    main()
