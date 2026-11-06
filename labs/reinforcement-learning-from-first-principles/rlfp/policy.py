"""A tiny autoregressive policy, so that every idea in the post can be checked exactly.

The post's claim in "A language model is already a policy" is that nothing has to be added to a language model to
make it an RL agent:

    state  s_t = (x, y_<t)      action a_t = y_t      policy pi(a_t | s_t) = softmax(logits)[a_t]
    transition s_{t+1} = s_t + a_t   (string concatenation: deterministic, known, free)

`TinyLM` is exactly that object, made small. It reads a token prefix and returns next-token logits. It uses a GRU
instead of a transformer because the transformer is not what this lab is about (C3 and C4 build one). The input
vocabulary can be larger than the action vocabulary, so a prompt can use tokens the policy never emits.
With a handful of actions and a short horizon, every possible completion can be listed, and expectations that the
post calls intractable become exact sums (see `exact.py`).
"""

from __future__ import annotations

import torch
from torch import nn


class TinyLM(nn.Module):
    """Embedding -> GRU -> linear head. `n_out=1` turns it into a critic with a scalar head."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.emb = nn.Embedding(n_in, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, tokens: torch.Tensor, h: torch.Tensor | None = None):
        """tokens (B, L) -> outputs (B, L, n_out), hidden state. outputs[:, i] reads tokens[:, : i + 1]."""
        out, h = self.rnn(self.emb(tokens), h)
        return self.head(out), h


def _teacher_forced_inputs(prompts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # Position Lp-1+t of the output predicts action t, so the input is the prompt plus all actions except the last.
    return torch.cat([prompts, actions[:, :-1]], dim=1)


def response_logits(model: TinyLM, prompts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """prompts (B, Lp), actions (B, T) -> logits (B, T, |A|) of pi(. | s_t) for t = 0 .. T-1."""
    out, _ = model(_teacher_forced_inputs(prompts, actions))
    return out[:, prompts.shape[1] - 1:]


def token_logprobs(model: TinyLM, prompts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """log pi(a_t | s_t) for every taken action: (B, T). Gather the taken token, discard the rest."""
    logp = torch.log_softmax(response_logits(model, prompts, actions), dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def token_values(critic: TinyLM, prompts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """V(s_t) for t = 0 .. T-1, read off the same positions as the policy: (B, T)."""
    return response_logits(critic, prompts, actions).squeeze(-1)


@torch.no_grad()
def sample(model: TinyLM, prompts: torch.Tensor, T: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Autoregressive sampling with the recurrent state carried forward (the GRU's version of a KV cache)."""
    out, h = model(prompts)
    actions = []
    for _ in range(T):
        probs = torch.softmax(out[:, -1].float(), dim=-1)
        a = torch.multinomial(probs, 1, generator=generator)  # (B, 1)
        actions.append(a)
        out, h = model(a + 0, h)  # actions share ids 0..|A|-1 with the input vocabulary
    return torch.cat(actions, dim=1)


def prompt_batch(prompt_ids: list[int] | torch.Tensor, repeat: int = 1) -> torch.Tensor:
    """One-token prompts, each repeated `repeat` times, prompt-major: (len(ids) * repeat, 1)."""
    ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    return ids.repeat_interleave(repeat).unsqueeze(1)
