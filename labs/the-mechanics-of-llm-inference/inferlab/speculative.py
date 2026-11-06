"""Speculative decoding with the rejection rule of Leviathan et al. (2022) and Chen et al. (2023).

A cheap draft proposes gamma tokens. The target scores all gamma + 1 positions in one pass. Token i is kept with
probability min(1, p_i(x) / q_i(x)). At the first rejection a replacement is drawn from max(p - q, 0), renormalised.
If all gamma are kept, one bonus token is drawn from the target's last distribution. The output follows the
target's distribution exactly; `toy_speculative_samples` lets a test check that by counting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .decoder import Decoder


def probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Next-token distribution. temperature 0 means greedy, written as a one-hot distribution."""
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits / temperature, dim=-1)


def accept_or_resample(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor,
                       gen: torch.Generator | None) -> tuple[list[int], int]:
    """p (gamma+1, V) target, q (gamma, V) draft, draft (gamma,) proposed ids -> (emitted ids, number accepted)."""
    out: list[int] = []
    for i, x in enumerate(draft.tolist()):
        r = torch.rand((), generator=gen, device=p.device)
        if r < torch.clamp(p[i, x] / q[i, x], max=1.0):
            out.append(x)
            continue
        residual = torch.clamp(p[i] - q[i], min=0.0)
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=gen)))
        return out, i
    out.append(int(torch.multinomial(p[-1], 1, generator=gen)))  # every proposal kept: a free bonus token
    return out, len(out) - 1


def toy_speculative_samples(p: torch.Tensor, q: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n independent one-token speculative draws for fixed distributions p (target) and q (draft)."""
    x = torch.multinomial(q, n, replacement=True, generator=gen)
    keep = torch.rand(n, generator=gen) < torch.clamp(p[x] / q[x], max=1.0)
    residual = torch.clamp(p - q, min=0.0)
    fallback = torch.multinomial(residual / residual.sum(), n, replacement=True, generator=gen)
    return torch.where(keep, x, fallback)


@dataclass
class SpecStats:
    tokens: list[int] = field(default_factory=list)
    target_passes: int = 0
    draft_passes: int = 0
    accepted: int = 0  # draft tokens kept
    tested: int = 0  # draft tokens that faced the acceptance test
    seconds: float = 0.0

    @property
    def alpha(self) -> float:
        """Per-token acceptance rate, counting only tokens that were actually tested."""
        return self.accepted / max(self.tested, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return len(self.tokens) / max(self.target_passes, 1)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def speculative_generate(target: Decoder, draft: Decoder, prompt: torch.Tensor, n_new: int, gamma: int = 4,
                         temperature: float = 0.0, seed: int = 0) -> SpecStats:
    """Generate n_new tokens for one sequence (prompt: (1, T)). Both models must share a vocabulary.

    Invariant: each cache holds every committed token except the newest one, which is fed next. Rejected
    positions are not erased: they are simply overwritten, because attention only reads up to the write position.
    """
    dev = prompt.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    total = prompt.shape[1] + n_new + gamma + 2
    target.allocate_cache(1, total)
    draft.allocate_cache(1, total)
    ids = prompt[0].tolist()
    st = SpecStats()
    _sync(dev)
    t0 = time.perf_counter()
    if len(ids) > 1:
        head = prompt[:, :-1]
        target.forward(head, 0)
        draft.forward(head, 0)
    draft_len = len(ids) - 1  # positions valid in the draft cache
    while len(st.tokens) < n_new:
        n = len(ids)
        # Draft catches up on tokens it has not seen (after a fully accepted round), then proposes gamma.
        feed = torch.tensor([ids[draft_len:]], device=dev)
        q_rows, proposals = [], []
        logits = draft.forward(feed, draft_len)
        st.draft_passes += 1
        for j in range(gamma):
            q = probs(logits[0, -1], temperature)
            x = int(torch.multinomial(q, 1, generator=gen)) if temperature > 0 else int(q.argmax())
            q_rows.append(q)
            proposals.append(x)
            if j < gamma - 1:
                logits = draft.forward(torch.tensor([[x]], device=dev), n + j)
                st.draft_passes += 1
        draft_len = n + gamma - 1
        # Target scores the newest committed token plus all proposals in one pass.
        verify = torch.tensor([[ids[-1]] + proposals], device=dev)
        p = probs(target.forward(verify, n - 1, last_only=False)[0], temperature)
        st.target_passes += 1
        emitted, k = accept_or_resample(p, torch.stack(q_rows), torch.tensor(proposals), gen)
        st.accepted += k
        st.tested += k + (1 if k < gamma else 0)
        ids += emitted
        st.tokens += emitted
        draft_len = min(draft_len, n + k)  # entries past the last kept proposal are stale
    _sync(dev)
    st.seconds = time.perf_counter() - t0
    st.tokens = st.tokens[:n_new]
    return st


@torch.no_grad()
def plain_generate(target: Decoder, prompt: torch.Tensor, n_new: int, temperature: float = 0.0,
                   seed: int = 0) -> SpecStats:
    """The baseline: one target pass per token."""
    dev = prompt.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    target.allocate_cache(1, prompt.shape[1] + n_new + 1)
    st = SpecStats()
    _sync(dev)
    t0 = time.perf_counter()
    logits = target.forward(prompt, 0)
    pos = prompt.shape[1]
    for _ in range(n_new):
        pr = probs(logits[0, -1], temperature)
        x = int(torch.multinomial(pr, 1, generator=gen)) if temperature > 0 else int(pr.argmax())
        st.tokens.append(x)
        st.target_passes += 1  # the prefill counts as the pass that produced the first token
        if len(st.tokens) < n_new:
            logits = target.forward(torch.tensor([[x]], device=dev), pos)
            pos += 1
    _sync(dev)
    st.seconds = time.perf_counter() - t0
    return st
