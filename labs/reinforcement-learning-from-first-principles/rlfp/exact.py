"""Exact expectations by listing every completion.

The post writes J(theta) = sum over tau of pi(tau) R(tau) and then says the sum is intractable: |V|^T terms. With
|A| = 6 actions and T = 5 tokens there are only 7,776 completions, so the sum can be done. That gives ground truth
for everything a sampled estimator only approximates: the true gradient, the true values V(s_t), the true
advantages, the exact entropy, the exact KL between two policies and the exact pass@k.
"""

from __future__ import annotations

import itertools

import torch

from .policy import TinyLM, token_logprobs


def all_sequences(n_actions: int, T: int) -> torch.Tensor:
    """Every completion of length T, in lexicographic order: (n_actions ** T, T)."""
    return torch.tensor(list(itertools.product(range(n_actions), repeat=T)), dtype=torch.long)


def seq_logprobs(model: TinyLM, prompt_id: int, seqs: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """log pi(tau) = sum_t log pi(a_t | s_t) for every listed completion: (S,). Keeps the autograd graph."""
    parts = []
    for i in range(0, len(seqs), chunk):
        s = seqs[i:i + chunk]
        prompts = torch.full((len(s), 1), prompt_id, dtype=torch.long)
        parts.append(token_logprobs(model, prompts, s).sum(-1))
    return torch.cat(parts)


def exact_J(logp_seq: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """J = sum_tau pi(tau) R(tau). Differentiable in theta through logp_seq: this is the exact gradient's source."""
    return (logp_seq.exp() * rewards).sum()


def entropy(logp_seq: torch.Tensor) -> torch.Tensor:
    """Entropy of the whole-completion distribution, in nats."""
    p = logp_seq.exp()
    return -(p * logp_seq).sum()


def kl(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) over completions."""
    return (logp_p.exp() * (logp_p - logp_q)).sum()


def pass_at_k(p_success: torch.Tensor | float, k: int) -> torch.Tensor:
    """Probability that at least one of k independent samples is correct: 1 - (1 - p)^k."""
    return 1 - (1 - torch.as_tensor(p_success, dtype=torch.float64)) ** k


def prefix_values(logp_seq: torch.Tensor, rewards: torch.Tensor, seqs: torch.Tensor, n_actions: int) -> torch.Tensor:
    """V(s_t) = E[R | prefix a_<t] for every listed completion and every t: (S, T + 1).

    Column t is the value of the state the completion was in before emitting token t. Column T is R itself, the
    value of the finished completion. With gamma = 1 and only a terminal reward, Q(s_t, a_t) = V(s_{t+1}), so the
    exact advantage of token t is column t+1 minus column t.
    """
    S, T = seqs.shape
    p = logp_seq.detach().exp().double()
    R = rewards.double()
    cols = []
    code = torch.zeros(S, dtype=torch.long)
    for t in range(T + 1):
        n_prefix = n_actions ** t
        num = torch.zeros(n_prefix, dtype=torch.float64).index_add_(0, code, p * R)
        den = torch.zeros(n_prefix, dtype=torch.float64).index_add_(0, code, p)
        cols.append((num / den.clamp(min=1e-300))[code])
        if t < T:
            code = code * n_actions + seqs[:, t]
    return torch.stack(cols, dim=1)


def exact_advantages(values: torch.Tensor) -> torch.Tensor:
    """A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = V(s_{t+1}) - V(s_t) for a deterministic, terminal-reward MDP: (S, T)."""
    return values[:, 1:] - values[:, :-1]


class ExactTask:
    """A set of one-token prompts with a reward table over every completion, and exact metrics for a policy."""

    def __init__(self, n_actions: int, T: int, reward_fn, prompt_ids: list[int], prompt_args: list):
        self.n_actions, self.T = n_actions, T
        self.seqs = all_sequences(n_actions, T)
        self.prompt_ids = prompt_ids
        self.prompt_args = prompt_args
        self.reward_fn = reward_fn
        self.rewards = [reward_fn(self.seqs, a) for a in prompt_args]  # one (S,) table per prompt

    def logps(self, model: TinyLM) -> list[torch.Tensor]:
        return [seq_logprobs(model, pid, self.seqs) for pid in self.prompt_ids]

    @torch.no_grad()
    def metrics(self, model: TinyLM, ks=(1, 16)) -> dict[str, float]:
        logps = self.logps(model)
        p = torch.stack([exact_J(lp, r) for lp, r in zip(logps, self.rewards)]).double()
        out = {f"pass@{k}": float(pass_at_k(p, k).mean()) for k in ks}
        out["seq_entropy"] = float(torch.stack([entropy(lp) for lp in logps]).mean())
        out["per_prompt_pass@1"] = [round(float(x), 4) for x in p]
        return out
