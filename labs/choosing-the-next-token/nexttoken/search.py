"""Greedy decoding, beam search with length normalization, and exhaustive search on a toy tree.

A "stepper" is any callable `step(prefixes, parents) -> logprobs (B, V)`:
    prefixes : the generated tokens of each live hypothesis (list of lists)
    parents  : for each row, which row of the previous call it extends (None on the first call)
The toy tree ignores `parents`. The Hugging Face stepper uses it to reorder its KV cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


class ToyTreeLM:
    """A language model written as a table: prefix tuple -> next-token probabilities. Unlisted prefixes emit EOS."""

    def __init__(self, vocab: list[str], table: dict[tuple, dict[str, float]], eos: str = "<eos>"):
        self.vocab, self.table, self.eos = vocab, table, eos
        self.index = {t: i for i, t in enumerate(vocab)}
        self.eos_id = self.index[eos]

    def probs(self, prefix: list[int]) -> torch.Tensor:
        row = self.table.get(tuple(self.vocab[i] for i in prefix), {self.eos: 1.0})
        p = torch.zeros(len(self.vocab))
        for t, v in row.items():
            p[self.index[t]] = v
        return p

    def __call__(self, prefixes, parents=None) -> torch.Tensor:
        return torch.stack([self.probs(p) for p in prefixes]).log()


def post_example_tree() -> ToyTreeLM:
    """The two-step example from the post: A is likelier first (0.6), B leads to a confident continuation (0.9)."""
    vocab = ["A", "B", "x", "y", "z", "<eos>"]
    table = {
        (): {"A": 0.6, "B": 0.4},
        ("A",): {"x": 0.35, "y": 0.33, "z": 0.32},
        ("B",): {"x": 0.90, "y": 0.05, "z": 0.05},
    }
    return ToyTreeLM(vocab, table)


class HFStepper:
    """Beam stepper over a Hugging Face model with a KV cache that is reordered to follow the surviving beams."""

    def __init__(self, model, prompt_ids: torch.Tensor):
        self.model, self.prompt_ids, self.cache = model, prompt_ids, None

    @torch.no_grad()
    def __call__(self, prefixes, parents=None) -> torch.Tensor:
        if self.cache is None:
            self.cache = DynamicCache(config=self.model.config)
            inp = self.prompt_ids
        else:
            self.cache.reorder_cache(torch.tensor(parents, device=self.model.device))
            inp = torch.tensor([[p[-1]] for p in prefixes])
        logits = self.model(input_ids=inp.to(self.model.device), past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1).logits[:, -1, :]
        return torch.log_softmax(logits.float().cpu(), dim=-1)


def length_penalty(length: int, alpha: float) -> float:
    """GNMT: lp(L) = ((5 + L) / 6) ** alpha. alpha = 0 gives raw log-probability."""
    return ((5 + length) / 6) ** alpha


@dataclass
class Hypothesis:
    tokens: list[int]
    logprob: float
    finished: bool = False
    score: float = field(default=0.0)


def greedy(step, eos_id: int, max_len: int) -> Hypothesis:
    tokens, total = [], 0.0
    parents = None
    for _ in range(max_len):
        logp = step([tokens], parents)[0]
        t = int(logp.argmax())
        tokens, total, parents = tokens + [t], total + float(logp[t]), [0]
        if t == eos_id:
            return Hypothesis(tokens, total, True)
    return Hypothesis(tokens, total, False)


def beam_search(step, eos_id: int, k: int, max_len: int, alpha: float = 0.0) -> list[Hypothesis]:
    """Keep k live hypotheses. Each step: expand every live beam by every token, keep the best k by log-probability.

    A candidate ending in EOS moves to the finished set and the live set is backfilled from the next-best
    candidates, so k beams stay live. Final ranking: logprob / lp(len), over finished hypotheses only. Unfinished
    beams are returned (after the finished ones) only if nothing finished. Returns hypotheses, best first.
    """
    live = [Hypothesis([], 0.0)]
    finished: list[Hypothesis] = []
    parents = None
    for t in range(max_len):
        logp = step([h.tokens for h in live], parents)  # (B, V)
        cand = torch.tensor([h.logprob for h in live]).unsqueeze(-1) + logp
        flat = cand.flatten()
        order = torch.argsort(flat, descending=True)[: 2 * k + 1]
        new_live, parents = [], []
        V = logp.shape[-1]
        for idx in order.tolist():
            b, tok = divmod(idx, V)
            score = float(flat[idx])
            if score == -math.inf:
                break
            hyp = Hypothesis(live[b].tokens + [tok], score)
            if tok == eos_id:
                hyp.finished = True
                finished.append(hyp)
            elif len(new_live) < k:
                new_live.append(hyp)
                parents.append(b)
            if len(new_live) == k and len(finished) >= k:
                break
        live = new_live
        if not live:
            break
        # With alpha = 0 scores only fall, so once k finished beams beat every live beam the search is over.
        if alpha == 0 and len(finished) >= k:
            kth = sorted(h.logprob for h in finished)[-k]
            if max(h.logprob for h in live) < kth:
                break
    for h in finished + live:
        h.score = h.logprob / length_penalty(len(h.tokens), alpha)
    rank = lambda hs: sorted(hs, key=lambda h: h.score, reverse=True)  # noqa: E731
    return rank(finished) + rank(live)


def exhaustive(lm: ToyTreeLM, max_len: int) -> list[Hypothesis]:
    """Every complete sequence of the toy tree with its probability, best first."""
    out = []

    def walk(prefix, lp):
        p = lm.probs(prefix)
        for i in torch.nonzero(p).flatten().tolist():
            nlp = lp + math.log(p[i])
            if i == lm.eos_id or len(prefix) + 1 == max_len:
                out.append(Hypothesis(prefix + [i], nlp, i == lm.eos_id))
            else:
                walk(prefix + [i], nlp)

    walk([], 0.0)
    return sorted(out, key=lambda h: h.logprob, reverse=True)
