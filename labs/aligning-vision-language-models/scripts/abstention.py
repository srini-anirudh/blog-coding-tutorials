"""Knowing when not to answer (post Figure 10): coverage, risk and the threshold c / (1 + c).

    python labs/aligning-vision-language-models/scripts/abstention.py

Question: "What colour is the <object>?" Image quality varies (noise sd 0.3 to 1.5). In a quarter of the test
questions the object's evidence has been cropped away, so the question cannot be answered from the image.
Two small classifiers read (image, question):
  overconfident    trained on answerable questions only, 4 colour classes
  trained to abstain   also sees cropped questions, labelled with a fifth class "can't tell"
Confidence is the probability of the top colour. Answer when confidence > tau. Score +1 right, -c wrong, 0 abstain.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from groundpref.metrics import expected_calibration_error, risk_coverage  # noqa: E402
from groundpref.report import Log  # noqa: E402
from groundpref.train import set_threads  # noqa: E402
from groundpref.world import COLOURS, F, K, OBJECTS, crop, render, sample_scene  # noqa: E402

ABSTAIN = len(COLOURS)


def data(n: int, seed: int, p_unanswerable: float):
    rng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    x, q, y, answerable = [], [], [], []
    for _ in range(n):
        s = sample_scene(rng)
        obj = rng.choice(list(s.facts))
        noise = rng.uniform(0.3, 1.5)
        img = render(s, gen, noise)
        k = OBJECTS.index(obj)
        if rng.random() < p_unanswerable:
            img = crop(img, [k], gen, noise)
            y.append(ABSTAIN)
            answerable.append(False)
        else:
            y.append(COLOURS.index(s.facts[obj].colour))
            answerable.append(True)
        x.append(img.flatten())
        q.append(k)
    return torch.stack(x), torch.tensor(q), torch.tensor(y), torch.tensor(answerable)


class Answerer(nn.Module):
    def __init__(self, classes: int, d: int = 128):
        super().__init__()
        self.q = nn.Embedding(K, d)
        self.net = nn.Sequential(nn.Linear(K * F + d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, classes))

    def forward(self, x, q):
        return self.net(torch.cat([x, self.q(q)], -1))


def fit(x, q, y, classes, steps=1500, seed=0):
    torch.manual_seed(seed)
    m = Answerer(classes)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    for _ in range(steps):
        j = torch.randint(len(y), (256,))
        loss = nn.functional.cross_entropy(m(x[j], q[j]), y[j])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=150000)
    ap.add_argument("--test", type=int, default=8000)
    args = ap.parse_args()
    set_threads()
    log = Log("abstention")
    xa, qa, ya, _ = data(args.train, 1, 0.0)
    xb, qb, yb, _ = data(args.train, 2, 0.25)
    xt, qt, yt, at = data(args.test, 3, 0.25)
    models = {"overconfident": fit(xa, qa, ya, len(COLOURS)), "trained to abstain": fit(xb, qb, yb, len(COLOURS) + 1)}
    taus = [i / 100 for i in range(100)]
    out = {}
    for name, m in models.items():
        with torch.no_grad():
            p = m(xt, qt).softmax(-1)[:, :len(COLOURS)]
        conf, pred = p.max(-1)
        correct = at & (pred == yt)
        r = {"mean_conf_answerable": conf[at].mean().item(), "mean_conf_unanswerable": conf[~at].mean().item(),
             "acc_answerable": correct[at].float().mean().item(),
             "ece_all": expected_calibration_error(conf, correct)}
        for c in (1.0, 4.0):
            curve = [risk_coverage(conf, correct, t, c) for t in taus]
            best = max(curve, key=lambda d: d["eff_reliability"])
            theory = risk_coverage(conf, correct, c / (1 + c), c)
            always = risk_coverage(conf, correct, -1.0, c)
            r[f"c={c:g}"] = {"best": best, "at_c_over_1_plus_c": theory, "always_answer": always}
        safe = [d for d in (risk_coverage(conf, correct, t, 1.0) for t in taus) if d["risk"] <= 0.05 and d["coverage"] > 0]
        r["coverage_at_risk_5pct"] = max((d["coverage"] for d in safe), default=0.0)
        out[name] = r
        log(f"\n{name}: accuracy on answerable {r['acc_answerable']:.3f}; mean confidence answerable "
            f"{r['mean_conf_answerable']:.3f}, unanswerable {r['mean_conf_unanswerable']:.3f}; ECE {r['ece_all']:.3f}; "
            f"coverage at risk <= 5%: {r['coverage_at_risk_5pct']:.3f}")
        for c in (1.0, 4.0):
            d = r[f"c={c:g}"]
            b, t, a = d["best"], d["at_c_over_1_plus_c"], d["always_answer"]
            log(f"  c={c:g}: always answer ER={a['eff_reliability']:+.3f} | best tau={b['tau']:.2f} "
                f"ER={b['eff_reliability']:+.3f} cov={b['coverage']:.3f} risk={b['risk']:.3f} | "
                f"tau=c/(1+c)={c / (1 + c):.2f} ER={t['eff_reliability']:+.3f} cov={t['coverage']:.3f} risk={t['risk']:.3f}")
    log.save({"args": vars(args), "results": out})


if __name__ == "__main__":
    main()
