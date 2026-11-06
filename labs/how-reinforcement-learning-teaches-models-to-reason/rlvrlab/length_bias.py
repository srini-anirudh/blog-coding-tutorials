"""Section 7 of the post: GRPO's 1/|o_i| normalization makes wrong answers longer.

A toy policy that only decides how long to write. Before it writes, the environment draws a hidden state the
policy can see: "knows" (the answer will be right with probability 0.9) or "guesses" (right with probability 0.1).
This stands in for a model that can tell, a few tokens in, whether it is on a good path. In each state the policy
has one learnable stop logit. At every token it stops with probability sigmoid(stop_logit[state]).

Length never changes correctness here, by construction. So any change in length is produced by the estimator,
not by a benefit of thinking longer.

Each response is |o| = L tokens (L-1 "continue" decisions and one "stop"), capped at `max_len`. The surrogate at
ratio 1 is  sum_i sum_t w_i * A_i * log pi(o_i,t),  with the per-token weight w_i set by the loss type:

    grpo:   w_i = 1 / (G * |o_i|)     z-scored advantage   (DeepSeekMath)
    dapo:   w_i = 1 / sum_j |o_j|     z-scored advantage   (token-level loss over the group, the DAPO formula)
    dapo_batch: w_i = P / sum over the whole batch of |o_j|  (token-level loss over the batch, as most trainers do)
    drgrpo: w_i = 1 / (G * max_len)   mean-centred advantage (Dr. GRPO)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .advantages import group_advantages

LOSSES = ("grpo", "dapo", "dapo_batch", "drgrpo")


@dataclass
class LengthConfig:
    loss: str = "grpo"
    steps: int = 400
    prompts: int = 64
    group: int = 8
    p_knows: float = 0.5
    acc_knows: float = 0.9
    acc_guess: float = 0.1
    init_len: float = 40.0  # starting mean length in both states
    max_len: int = 1024
    lr: float = 0.02
    seed: int = 0


def sample_lengths(stop_logit: torch.Tensor, state: torch.Tensor, max_len: int, gen: torch.Generator):
    """Geometric lengths with stop probability sigmoid(stop_logit[state]), capped at max_len."""
    q = torch.sigmoid(stop_logit.detach())[state]
    u = torch.rand(state.shape, generator=gen).clamp_min(1e-12)
    length = torch.floor(torch.log(u) / torch.log1p(-q)) + 1
    truncated = length > max_len
    return length.clamp(max=max_len), truncated


def response_logprob(stop_logit, state, length, truncated):
    """log pi(o) = (L-1) log(1-q) + log q, or L log(1-q) when the cap cut the response off."""
    z = stop_logit[state]
    log_q, log_1mq = torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)
    n_continue = torch.where(truncated, length, length - 1)
    return n_continue * log_1mq + (~truncated) * log_q


def loss_weights(length: torch.Tensor, loss: str, max_len: int) -> torch.Tensor:
    g = length.shape[-1]
    if loss == "grpo":
        return 1.0 / (g * length)
    if loss == "dapo":
        return (1.0 / length.sum(-1, keepdim=True)).expand_as(length)
    if loss == "dapo_batch":
        return torch.full_like(length, length.shape[0] / float(length.sum()))
    if loss == "drgrpo":
        return torch.full_like(length, 1.0 / (g * max_len))
    raise ValueError(loss)


def train(cfg: LengthConfig) -> list[dict]:
    """Run the toy RL loop. Returns one log row every 10 steps."""
    gen = torch.Generator().manual_seed(cfg.seed)
    q0 = 1.0 / cfg.init_len
    stop_logit = torch.full((2,), float(torch.logit(torch.tensor(q0))), requires_grad=True)  # [knows, guesses]
    opt = torch.optim.Adam([stop_logit], lr=cfg.lr)
    acc = torch.tensor([cfg.acc_knows, cfg.acc_guess])
    log = []
    for step in range(cfg.steps + 1):
        state = (torch.rand(cfg.prompts, cfg.group, generator=gen) > cfg.p_knows).long()  # 0 knows, 1 guesses
        length, truncated = sample_lengths(stop_logit, state, cfg.max_len, gen)
        reward = (torch.rand(state.shape, generator=gen) < acc[state]).float()
        adv = group_advantages(reward, "mean" if cfg.loss == "drgrpo" else "grpo")
        w = loss_weights(length, cfg.loss, cfg.max_len)
        if step % 10 == 0:
            wrong = reward == 0
            log.append({
                "step": step,
                "len_wrong": float(length[wrong].mean()),
                "len_right": float(length[~wrong].mean()),
                "len_knows": float(1 / torch.sigmoid(stop_logit[0].detach())),
                "len_guess": float(1 / torch.sigmoid(stop_logit[1].detach())),
                "accuracy": float(reward.mean()),
                "truncated": float(truncated.float().mean()),
            })
        if step == cfg.steps:
            break
        lp = response_logprob(stop_logit, state, length, truncated)
        loss = -(w * adv * lp).sum() / cfg.prompts
        opt.zero_grad()
        loss.backward()
        opt.step()
    return log


def stop_gradient_bias(cfg: LengthConfig, groups: int = 200_000) -> dict[str, float]:
    """Mean ascent direction on each state's stop logit at initialization, with its Monte Carlo standard error.

    Negative means "write longer". No optimizer is involved, so this is the estimator's own bias.
    The per-sample gradient is w_i * A_i * d log pi(o_i)/dz, with d log pi/dz = (1 - q) - (L - 1) q.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    z = torch.full((2,), float(torch.logit(torch.tensor(1.0 / cfg.init_len))))
    state = (torch.rand(groups, cfg.group, generator=gen) > cfg.p_knows).long()
    length, truncated = sample_lengths(z, state, cfg.max_len, gen)
    acc = torch.tensor([cfg.acc_knows, cfg.acc_guess])
    reward = (torch.rand(state.shape, generator=gen) < acc[state]).float()
    adv = group_advantages(reward, "mean" if cfg.loss == "drgrpo" else "grpo")
    q = torch.sigmoid(z)[state]
    dlp = torch.where(truncated, length, length - 1) * (-q) + (~truncated) * (1 - q)
    if cfg.loss == "dapo_batch":  # one batch of `prompts` groups shares the denominator
        lb = length.view(-1, cfg.prompts, cfg.group)
        w = (cfg.prompts / lb.sum(dim=(1, 2), keepdim=True)).expand_as(lb).reshape(length.shape)
    else:
        w = loss_weights(length, cfg.loss, cfg.max_len)
    contrib = w * adv * dlp
    out = {}
    for s, name in ((0, "knows"), (1, "guess")):
        per = (contrib * (state == s)).sum(-1)
        out[name] = float(per.mean())
        out[name + "_stderr"] = float(per.std() / groups**0.5)
        out[name + "_snr"] = float(per.mean() / per.std())  # bias per group, in units of one group's noise
    return out


def expected_stop_gradient(adv_wrong: float, adv_right: float, q: float, loss: str, mean_len: float) -> dict:
    """Sign of the per-response push on the stop logit, for one wrong and one right response of length L.

    d/dz log pi(o) = (1 - q) - (L - 1) q = 1 - qL. Under grpo the weight is A/L, so the push is A (1/L - q).
    Averaged over geometric L, E[1/L] > q (Jensen), so a negative A lowers the stop logit: wrong gets longer.
    Under drgrpo the weight does not depend on L, and E[1 - qL] = 0: no push at all.
    """
    ls = torch.arange(1, 20_000, dtype=torch.float64)
    pmf = q * (1 - q) ** (ls - 1)
    score = 1 - q * ls
    if loss == "grpo":
        push = (pmf * score / ls).sum()
    else:
        push = (pmf * score).sum() / mean_len
    return {"wrong": float(adv_wrong * push), "right": float(adv_right * push)}
