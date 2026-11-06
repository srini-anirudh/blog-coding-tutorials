"""The policy-gradient estimator and its baselines.

    grad J = E_tau [ (R(tau) - b) * sum_t grad log pi(a_t | s_t) ]     (REINFORCE, Williams 1992)

`pg_loss` is written so that its gradient is exactly that estimate: the loss is minus the weighted sum of the taken
tokens' log-probabilities, summed over tokens (log pi(tau) is a sum) and averaged over samples. Read as the post
puts it: supervised fine-tuning on your own samples, with each sample's loss weighted by its reward.
"""

from __future__ import annotations

import torch


def pg_loss(token_logp: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """token_logp (N, T) with grad. weights (N,) per sample or (N, T) per token, no grad. Returns a scalar."""
    w = weights.detach()
    if w.dim() == 1:
        w = w.unsqueeze(1)
    per_token = w * token_logp
    if mask is not None:
        per_token = per_token * mask.to(per_token.dtype)
    return -per_token.sum(-1).mean()


def baseline_weights(rewards: torch.Tensor, kind: str, group_size: int = 1) -> torch.Tensor:
    """Per-sample weights R - b for one batch laid out prompt-major (all G samples of prompt 0 first).

    none:       b = 0                                  plain REINFORCE
    batch:      b = mean of the whole batch, including this sample
    loo:        b = mean of the other samples in the batch (leave-one-out)
    group:      b = mean of the other G-1 samples of the same prompt (RLOO)
    group_std:  (R - group mean) / group std           (GRPO's advantage; the division is not a baseline)
    """
    R = rewards.float()
    if kind == "none":
        return R
    if kind == "batch":
        return R - R.mean()
    if kind == "loo":
        n = R.numel()
        return R - (R.sum() - R) / (n - 1)
    g = R.view(-1, group_size)
    if kind == "group":
        return (g - (g.sum(1, keepdim=True) - g) / (group_size - 1)).reshape(-1)
    if kind == "group_std":
        return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).reshape(-1)
    raise ValueError(kind)


def flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool = False) -> torch.Tensor:
    """d loss / d params as one vector."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def error_stats(estimates: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """estimates (K, D) independent gradient estimates of `truth` (D,).

    mse:      E || g - grad J ||^2, the error of a single estimate
    bias_sq:  || mean(g) - grad J ||^2, which shrinks like mse / K if the estimator is unbiased
    snr:      || grad J || / sqrt(mse), signal over noise in amplitude
    cos:      mean cosine between an estimate and the true gradient
    """
    err = estimates - truth
    mse = err.pow(2).sum(-1).mean()
    cos = (estimates @ truth) / (estimates.norm(dim=-1).clamp(min=1e-30) * truth.norm().clamp(min=1e-30))
    return {
        "mse": float(mse),
        "bias_sq": float((estimates.mean(0) - truth).pow(2).sum()),
        "snr": float(truth.norm() / mse.sqrt().clamp(min=1e-30)),
        "cos": float(cos.mean()),
        "true_norm_sq": float(truth.pow(2).sum()),
    }
