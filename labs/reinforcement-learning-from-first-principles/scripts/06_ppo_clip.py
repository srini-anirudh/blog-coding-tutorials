"""Section 10 (and section 14, point 2): PPO's clipped objective.

1. The four regions, by autograd: the value of min(r A, clip(r) A) and its derivative in r.
2. Reuse one batch for 16 epochs, with and without the clip. Each epoch reports the share of tokens whose ratio
   is outside [1 - eps, 1 + eps], the largest ratio, and the exact KL(pi || pi_old) over all completions. The clip
   slows the drift. It does not bound it.
3. Ordinary PPO training (4 epochs per batch). How often does the clip actually fire?

    python labs/reinforcement-learning-from-first-principles/scripts/06_ppo_clip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.estimators import baseline_weights  # noqa: E402
from rlfp.exact import kl  # noqa: E402
from rlfp.policy import token_logprobs  # noqa: E402
from rlfp.ppo import clip_region_table, clipped_surrogate  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.train import TrainConfig, make_policy, rollout, sum_task, train  # noqa: E402


def reuse_one_batch(rep: Report, eps: float = 0.2, epochs: int = 16, lr: float = 1e-2, G: int = 16) -> None:
    task = sum_task()
    rep.print(f"\n2. One batch of {5 * G} rollouts reused for {epochs} epochs, Adam lr {lr}, eps {eps}")
    out = {}
    for clip in (True, False):
        policy = make_policy(task, hidden=64, seed=0)
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        prompts, acts, R = rollout(policy, task, G, torch.Generator().manual_seed(0))
        adv = baseline_weights(R, "group", G)
        mask = torch.ones_like(acts, dtype=torch.float32)
        with torch.no_grad():
            old = token_logprobs(policy, prompts, acts)
            old_full = [lp.clone() for lp in task.logps(policy)]
        rows = []
        for ep in range(1, epochs + 1):
            loss, st = clipped_surrogate(token_logprobs(policy, prompts, acts), old, adv, mask, eps, clip=clip)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                _, st = clipped_surrogate(token_logprobs(policy, prompts, acts), old, adv, mask, eps)
                k = float(torch.stack([kl(a, b) for a, b in zip(task.logps(policy), old_full)]).mean())
            if ep in (1, 2, 4, 8, 16):
                rows.append({"epoch": ep, "outside_frac": st["outside_frac"], "clip_frac": st["clip_frac"],
                             "ratio_max": st["ratio_max"], "ratio_min": st["ratio_min"], "exact_kl": k})
        name = "clipped" if clip else "unclipped"
        rep.print(f"{name} surrogate")
        rep.table(rows)
        out[name] = rows
    rep.data["reuse_one_batch"] = out


def clip_in_training(rep: Report) -> None:
    out = train(TrainConfig(method="ppo", steps=200, lr=3e-3, ppo_epochs=4, eval_every=50, track_step_kl=True))
    stats = [h for h in out["history"] if "clip_frac" in h]
    cf = torch.tensor([h["clip_frac"] for h in stats])
    of = torch.tensor([h["outside_frac"] for h in stats])
    final = [h for h in out["history"] if "pass@1" in h][-1]
    res = {"mean_clip_frac": float(cf.mean()), "max_clip_frac": float(cf.max()),
           "mean_outside_frac": float(of.mean()), "final_pass@1": final["pass@1"]}
    rep.print(f"\n3. PPO training, 200 steps x 4 epochs, lr 3e-3, eps 0.2, measured in the forward pass of the 4th epoch: "
              f"clipped tokens mean {res['mean_clip_frac']:.3f}, max {res['max_clip_frac']:.3f}; ratio outside "
              f"the band mean {res['mean_outside_frac']:.3f}; final pass@1 {res['final_pass@1']:.3f}")
    rep.data["clip_in_training"] = res


def main() -> None:
    rep = Report("06_ppo_clip")
    rep.print("1. The four regions of min(r A, clip(r, 0.8, 1.2) A)")
    rows = clip_region_table(0.2)
    rep.table(rows)
    rep.data["regions"] = rows
    reuse_one_batch(rep)
    clip_in_training(rep)
    rep.save()


if __name__ == "__main__":
    main()
