"""Sections 11 and 12: one RL step written out in shapes, then the masks, the normalisation and the memory.

1. A full PPO-style step on a toy policy with variable-length prompts, EOS and padding. Every tensor's shape is
   printed in the order of the post's listing.
2. The mask bug: forget to mask tokens after EOS. The run still trains; the gradient is quietly different.
3. Loss normalisation: the weight one token gets in a 100-token and a 2,000-token answer.
4. The logits tensor: the post's 268 GB, recomputed, and a chunked log-prob gather that never builds it.
5. The four models in the room, at 7B, with the post's bytes-per-parameter accounting.

    python labs/reinforcement-learning-from-first-principles/scripts/08_rl_step_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlfp.advantages import gae, returns, terminal_rewards  # noqa: E402
from rlfp.estimators import flat_grad  # noqa: E402
from rlfp.policy import TinyLM, response_logits, sample, token_values  # noqa: E402
from rlfp.ppo import clipped_surrogate, token_entropy  # noqa: E402
from rlfp.report import Report  # noqa: E402
from rlfp.tensors import (chunked_token_logprobs, logits_bytes, masked_loss, per_token_weight,  # noqa: E402
                          resident_gb, response_mask)

EOS, PAD = 10, 11  # actions are digits 0-9 and EOS; PAD only ever appears in prompts


def shapes(rep: Report) -> None:
    torch.manual_seed(0)
    B, G, T = 4, 4, 12
    policy, ref, critic = TinyLM(12, 11), TinyLM(12, 11), TinyLM(12, 1)
    ref.load_state_dict(policy.state_dict())
    raw = [[3, 1, 4], [1, 5], [9, 2, 6, 5], [3]]
    Lp = max(len(p) for p in raw)
    prompts = torch.tensor([[PAD] * (Lp - len(p)) + p for p in raw])  # left padding
    gen = torch.Generator().manual_seed(0)
    show = lambda name, t, note="": rep.print(f"{name:<12} {str(list(t.shape)):<14} {note}")  # noqa: E731
    show("prompts", prompts, "[B, Lp], left-padded")
    pr = prompts.repeat_interleave(G, 0)
    actions = sample(policy, pr, T, gen)
    show("sequences", torch.cat([pr, actions], 1), "[B*G, Lp + T] after the rollout")
    mask = response_mask(actions, EOS)
    lengths = mask.sum(1).long()
    target_len = torch.tensor([len(p) for p in raw]).repeat_interleave(G)
    R = (lengths - 1 == target_len).float()  # reward: answer as many digits as the prompt has, then EOS
    show("rewards", R, f"[B*G], mean {float(R.mean()):.3f}")
    logits = response_logits(policy, pr, actions)
    show("logits", logits, "[B*G, T, |V|]  <- the memory spike at real scale")
    logp = logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ref_logp = response_logits(ref, pr, actions).log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    show("logprobs", logp, "[B*G, T] after gathering the taken token (policy and reference)")
    values = token_values(critic, pr, actions)
    show("values", values, "[B*G, T]")
    r = terminal_rewards(R, T, lengths)
    adv = gae(r, values.detach(), 1.0, 0.95, mask)
    show("advantages", adv, "[B*G, T], one backward scan, zero after EOS")
    show("mask", mask, f"[B*G, T], {int(mask.sum())} of {mask.numel()} positions are real response tokens")
    pol_loss, st = clipped_surrogate(logp, logp.detach(), adv, mask)
    v_loss = masked_loss((values - returns(r)).pow(2), mask, "token")
    ent = token_entropy(logits, mask)
    kl = masked_loss(torch.exp(ref_logp - logp) - (ref_logp - logp) - 1, mask, "token")
    loss = pol_loss + 0.5 * v_loss - 0.01 * ent
    rep.print(f"loss = clip {pol_loss.item():.4f} + 0.5 * value {v_loss.item():.4f} - 0.01 * entropy "
              f"{ent.item():.4f}; ratio mean on the first epoch = 1 exactly ({st['ratio_max']:.1f} max); "
              f"KL to reference {kl.item():.1e} (policy == reference)")

    # 2. The mask bug.
    params = list(policy.parameters())
    good = flat_grad(clipped_surrogate(logp, logp.detach(), adv, mask)[0], params, retain_graph=True)
    adv_bad = gae(r, values.detach(), 1.0, 0.95)  # no mask in the scan either
    bad = flat_grad(clipped_surrogate(logp, logp.detach(), adv_bad, torch.ones_like(mask))[0], params)
    cos = float(good @ bad / (good.norm() * bad.norm()))
    rel = float((good - bad).norm() / good.norm())
    rep.print(f"\n2. Forgetting the post-EOS mask: cosine(correct grad, buggy grad) = {cos:.3f}, "
              f"relative difference = {rel:.3f}; {int((1 - mask).sum())} masked positions were included")
    rep.data["mask_bug"] = {"cos": cos, "rel_diff": rel}


def normalisation(rep: Report) -> None:
    seq = per_token_weight([100, 2000], "seq")
    tok = per_token_weight([100, 2000], "token")
    rep.print(f"\n3. Weight of one token, answers of 100 and 2000 tokens: sequence-mean {seq[0]:.2e} vs "
              f"{seq[1]:.2e} (ratio {seq[0] / seq[1]:.0f}x); token-mean {tok[0]:.2e} vs {tok[1]:.2e} "
              f"(ratio {tok[0] / tok[1]:.0f}x)")
    rep.data["normalisation"] = {"seq": seq, "token": tok}


def logits_memory(rep: Report) -> None:
    gb = logits_bytes(256, 4096, 128_000) / 1e9
    torch.manual_seed(0)
    hidden, head = torch.randn(8, 512, 64), torch.nn.Linear(64, 8000)
    targets = torch.randint(0, 8000, (8, 512))
    with torch.no_grad():
        full = head(hidden).log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        chunked = chunked_token_logprobs(hidden, head, targets, chunk=64)
    diff = float((full - chunked).abs().max())
    rep.print(f"\n4. Logits for B*G = 256, T = 4096, |V| = 128,000 in bf16: {gb:.1f} GB. Chunked gather on "
              f"[8, 512, 8000]: max |full - chunked| = {diff:.1e}; peak logits {8 * 512 * 8000 * 4 / 1e6:.0f} MB "
              f"full vs {8 * 64 * 8000 * 4 / 1e6:.0f} MB per chunk of 64 positions (fp32)")
    rep.data["logits"] = {"gb_bf16": gb, "chunk_max_diff": diff}


def four_models(rep: Report) -> None:
    rep.print("\n5. Weights and optimiser state at 7B parameters, GB (16 B/param trainable, 2 B/param frozen)")
    rows = []
    for name, comp in [("PPO with reward model", dict(policy=1, critic=1, reference=1, reward_model=1)),
                       ("+ separate inference copy", dict(policy=1, critic=1, reference=1, reward_model=1,
                                                          inference_copy=1)),
                       ("verifier instead of reward model", dict(policy=1, critic=1, reference=1)),
                       ("critic-free (group baseline)", dict(policy=1, reference=1)),
                       ("critic-free, no reference KL", dict(policy=1))]:
        g = resident_gb(7e9, comp)
        rows.append({"setup": name, **{k: g.get(k, 0.0) for k in
                                       ["policy", "critic", "reference", "reward_model", "inference_copy", "total"]}})
    rep.table(rows, fmt="{:.0f}")
    rep.data["four_models"] = rows


def main() -> None:
    rep = Report("08_rl_step_shapes")
    rep.print("1. One step, B = 4 prompts, G = 4 samples each, T = 12, |V| = 11 actions")
    shapes(rep)
    normalisation(rep)
    logits_memory(rep)
    four_models(rep)
    rep.save()


if __name__ == "__main__":
    main()
