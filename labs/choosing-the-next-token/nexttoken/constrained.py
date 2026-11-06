"""Constrained decoding: a character automaton, a per-state token mask built over the real vocabulary, and the
distortion that masking introduces.

The pattern is a list of segments, matched left to right:
    ("lit", text)                 the exact characters of `text`
    ("cls", chars, lo, hi)        between lo and hi characters drawn from `chars`
A state is (segment index, characters consumed inside that segment). It is deterministic as long as a class never
contains the first character of the literal after it, which holds for the pattern below.
"""

from __future__ import annotations

import itertools
import math
import string

import torch

LETTERS = string.ascii_letters + " "
DIGITS = string.digits

PERSON_PATTERN = [
    ("lit", '{"name": "'),
    ("cls", LETTERS, 1, 24),
    ("lit", '", "age": '),
    ("cls", DIGITS, 1, 3),
    ("lit", "}"),
]


class CharDFA:
    def __init__(self, segments):
        self.segments = segments
        self.start = (0, 0)
        self.accept = (len(segments), 0)

    def step(self, state, ch):
        """Advance by one character. Returns the new state, or None if the character is not allowed."""
        seg, n = state
        if seg == len(self.segments):
            return None
        kind = self.segments[seg][0]
        if kind == "lit":
            text = self.segments[seg][1]
            if ch != text[n]:
                return None
            return (seg + 1, 0) if n + 1 == len(text) else (seg, n + 1)
        _, chars, lo, hi = self.segments[seg]
        if ch in chars and n < hi:
            return (seg, n + 1)
        if n >= lo:  # the class may end here: try the next segment from its start
            return self.step((seg + 1, 0), ch)
        return None

    def walk(self, state, text: str):
        for ch in text:
            state = self.step(state, ch)
            if state is None:
                return None
        return state

    def matches(self, text: str) -> bool:
        """True if `text` is a complete match. The pattern must end in a literal, so the end state is exact."""
        return self.walk(self.start, text) == self.accept


class TokenIndex:
    """For each automaton state: which vocabulary tokens are allowed, and where each one leads.

    Built lazily, once per state, by walking every token string through the automaton. At decode time a step is a
    dictionary lookup, which is the point of Willard and Louf's precomputation.
    """

    def __init__(self, dfa: CharDFA, token_strings: list[str], eos_id: int):
        self.dfa, self.tokens, self.eos_id = dfa, token_strings, eos_id
        self.cache: dict = {}

    def allowed(self, state) -> dict[int, tuple]:
        if state not in self.cache:
            nxt = {}
            for i, s in enumerate(self.tokens):
                if i == self.eos_id or not s:
                    continue
                t = self.dfa.walk(state, s)
                if t is not None:
                    nxt[i] = t
            if state == self.dfa.accept:
                nxt[self.eos_id] = state
            self.cache[state] = nxt
        return self.cache[state]

    def mask(self, state, vocab_size: int) -> torch.Tensor:
        m = torch.zeros(vocab_size, dtype=torch.bool)
        m[list(self.allowed(state).keys())] = True
        return m


# --- The distortion: masking is not conditioning. Exact on a toy autoregressive model. ---

def toy_lm(vocab: int, length: int, seed: int = 0, concentration: float = 0.5):
    """A random autoregressive model: p(y_t | y_<t) is a Dirichlet draw for every prefix."""
    g = torch.Generator().manual_seed(seed)
    table = {}
    for t in range(length):
        for prefix in itertools.product(range(vocab), repeat=t):
            gamma = torch._standard_gamma(torch.full((vocab,), concentration), generator=g)
            table[prefix] = gamma / gamma.sum()
    return table


def seq_prob(table, y) -> float:
    return math.prod(float(table[tuple(y[:t])][y[t]]) for t in range(len(y)))


def exact_conditional(table, vocab, length, in_language) -> dict:
    """p(y | y in L) by enumeration."""
    seqs = [y for y in itertools.product(range(vocab), repeat=length) if in_language(y)]
    z = sum(seq_prob(table, y) for y in seqs)
    return {y: seq_prob(table, y) / z for y in seqs}


def masked_distribution(table, vocab, length, in_language) -> dict:
    """What step-wise masking samples: at each step renormalize over tokens from which L is still reachable.

    This is the best case for masking (a perfect 'can still complete' mask). It is still not p(y | L).
    """
    ok_prefixes = set()
    for y in itertools.product(range(vocab), repeat=length):
        if in_language(y):
            for t in range(length + 1):
                ok_prefixes.add(y[:t])
    out = {}
    for y in itertools.product(range(vocab), repeat=length):
        if not in_language(y):
            continue
        q = 1.0
        for t in range(length):
            p = table[y[:t]]
            allowed = [v for v in range(vocab) if y[:t] + (v,) in ok_prefixes]
            q *= float(p[y[t]]) / sum(float(p[v]) for v in allowed)
        out[y] = q
    return out


def total_variation(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def importance_resample(table, vocab, length, in_language, n_particles: int, n_rounds: int, seed: int = 0) -> dict:
    """Fix the distortion with importance weights: draw from the masked sampler q, weight by p(y) / q(y), resample.

    The weight equals the product of the allowed mass at each step. Returns the empirical distribution over
    n_rounds draws, each the result of resampling one of n_particles weighted particles.
    """
    q = masked_distribution(table, vocab, length, in_language)
    seqs = list(q)
    qv = torch.tensor([q[y] for y in seqs])
    w = torch.tensor([seq_prob(table, y) / q[y] for y in seqs])
    g = torch.Generator().manual_seed(seed)
    picks = torch.multinomial(qv, n_particles * n_rounds, replacement=True, generator=g).view(n_rounds, n_particles)
    chosen = torch.multinomial(w[picks], 1, generator=g).squeeze(-1)
    final = picks[torch.arange(n_rounds), chosen]
    counts = torch.bincount(final, minlength=len(seqs)).float() / n_rounds
    return {y: float(c) for y, c in zip(seqs, counts)}
