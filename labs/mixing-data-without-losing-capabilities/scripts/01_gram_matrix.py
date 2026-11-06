"""One inner product decides everything: measure K and check dL ~ -eta K p (post section 3).

Train the small model briefly on a uniform mix of five tasks. At two checkpoints, compute each task's gradient on
1,024 fixed examples, form the Gram matrix K_ij = <g_i, g_j>, and print the cosines. Then take one real SGD step
along sum_i p_i g_i for several mixtures p and step sizes, and compare each task's actual loss change with the
first-order prediction -eta (K p)_j. Finally, apply PCGrad to the most conflicting pair.

    python labs/mixing-data-without-losing-capabilities/scripts/01_gram_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab import grads as G  # noqa: E402
from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report  # noqa: E402
from mixlab.train import make_pools, train  # noqa: E402

TASKS = ("add", "sub", "rev", "sort", "copy")


def main() -> None:
    torch.set_num_threads(4)
    rep = Report("01_gram_matrix")
    tr, ev = make_pools(TASKS, n_train=20000, n_eval=1024)
    model = make_model(seed=0)
    done = 0
    for ckpt in (300, 1500):
        train(model, tr, {t: 1 for t in TASKS}, steps=ckpt - done, seed=ckpt)
        done = ckpt
        g = G.task_gradients(model, ev)
        vecs = [g[t] for t in TASKS]
        cos = G.cosine(vecs)
        K = G.gram(vecs)
        rep(f"== checkpoint after {ckpt} steps of a uniform 5-task mixture ==")
        rep("gradient norms: " + ", ".join(f"{t} {float(v.norm()):.3f}" for t, v in zip(TASKS, vecs)))
        rep("cosine between task gradients:")
        rep.table(["", *TASKS], [[a, *(f"{float(cos[i, j]):+.2f}" for j in range(5))] for i, a in enumerate(TASKS)])
        eig = torch.linalg.eigvalsh(K)
        rep(f"K eigenvalues (PSD check): min {float(eig.min()):.2e}, max {float(eig.max()):.2e}")
        rep("")
        rep("one SGD step along g = sum_i p_i g_i: actual dL_j vs predicted -eta (K p)_j")
        mixes = {"uniform": [0.2] * 5, **{f"only {t}": [1.0 if s == t else 0.0 for s in TASKS] for t in ("add", "rev")}}
        rows = []
        for name, p in mixes.items():
            d = sum(pi * v for pi, v in zip(p, vecs))
            for eta in (0.001, 0.01, 0.1, 1.0):
                res = G.one_step_check(model, ev, d, eta)
                act = [res[t][0] for t in TASKS]
                pred = [res[t][1] for t in TASKS]
                err = sum(abs(a - b) for a, b in zip(act, pred)) / max(1e-12, sum(abs(b) for b in pred))
                sum_act = sum(pi * a for pi, a in zip([0.2] * 5, act))
                rows.append([name, eta, *(f"{a:+.4f}/{b:+.4f}" for a, b in zip(act, pred)), f"{err:.2f}", f"{sum_act:+.4f}"])
                rep.data.setdefault(f"ckpt{ckpt}", {}).setdefault("steps", []).append(
                    {"mix": name, "eta": eta, "actual": act, "predicted": pred, "rel_err": err}
                )
        rep.table(["mixture", "eta", *(f"{t} act/pred" for t in TASKS), "rel err", "mean dL (actual)"], rows)
        # PCGrad on the most conflicting pair.
        i, j = min(((a, b) for a in range(5) for b in range(5) if a < b), key=lambda ab: float(cos[ab[0], ab[1]]))
        ga, gb = vecs[i], vecs[j]
        proj = G.pcgrad_pair(ga, gb)
        res_raw = G.one_step_check(model, ev, ga, 0.03)
        res_pc = G.one_step_check(model, ev, proj, 0.03)
        rep("")
        rep(f"most conflicting pair: {TASKS[i]} vs {TASKS[j]}, cos {float(cos[i, j]):+.3f}")
        rep(f"  <PCGrad(g_{TASKS[i]}), g_{TASKS[j]}> = {float(proj.dot(gb)):+.2e} (zero when they conflicted)")
        rep(f"  step on g_{TASKS[i]} (eta 0.03): d L_{TASKS[j]} = {res_raw[TASKS[j]][0]:+.5f}, d L_{TASKS[i]} = {res_raw[TASKS[i]][0]:+.5f}")
        rep(f"  step on PCGrad(g_{TASKS[i]}):    d L_{TASKS[j]} = {res_pc[TASKS[j]][0]:+.5f}, d L_{TASKS[i]} = {res_pc[TASKS[i]][0]:+.5f}")
        rep.data[f"ckpt{ckpt}"].update(
            {"cos": cos.tolist(), "norms": [float(v.norm()) for v in vecs], "pair": [TASKS[i], TASKS[j]]}
        )
        rep("")
    rep.save()


if __name__ == "__main__":
    main()
