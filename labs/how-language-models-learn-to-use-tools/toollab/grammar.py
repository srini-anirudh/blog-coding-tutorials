"""Constrained decoding: a schema becomes an automaton, the automaton becomes a mask on the vocabulary.

`call_mask` builds the mask for this lab's call syntax from the tools in context. At each step it
returns A_t, the set of tokens that keep the prefix inside the language. The decoder adds -inf to
every logit outside A_t (see model.decode).

`local_vs_global` works the post's subtlety on a distribution small enough to enumerate: locally
masked decoding is not the model's distribution conditioned on validity, and the missing factor is
alpha(prefix), the probability mass of valid continuations still reachable from the prefix.
"""

from __future__ import annotations

import math

from .world import TYPES, VALUES, Tool

ALL_VALUES = {v for t in TYPES for v in VALUES[t]}
VALUE = object()  # marker for "a value slot" in a skeleton


def skeleton(t: Tool) -> list:
    out: list = [*t.name, "("]
    for k, (fname, ftype) in enumerate(t.fields):
        if k:
            out.append(",")
        out += [*fname, "=", (VALUE, ftype)]
    return out + [")", "<eos>"]


def call_mask(tools: list[Tool], typed: bool = False):
    """Return mask(prefix) -> allowed next tokens. typed=True restricts each value to its field's type."""
    skels = [skeleton(t) for t in tools]

    def allowed_at(sk: list, i: int) -> set[str]:
        slot = sk[i]
        if isinstance(slot, tuple):
            return set(VALUES[slot[1]]) if typed else set(ALL_VALUES)
        return {slot}

    def mask(prefix: list[str]) -> set[str]:
        if not prefix:
            return {"NONE"} | {sk[0] for sk in skels}
        if prefix[0] == "NONE":
            return {"<eos>"}
        live = [sk for sk in skels if len(sk) > len(prefix)
                and all(p in allowed_at(sk, i) for i, p in enumerate(prefix))]
        return set().union(*(allowed_at(sk, len(prefix)) for sk in live)) if live else {"<eos>"}

    return mask


# ---- The subtlety, on a distribution we can enumerate exactly ----

def enumerate_strings(next_probs, prefix=(), stop="$"):
    """All complete strings of a finite autoregressive model, with their probabilities."""
    out = {}
    for tok, p in next_probs(prefix).items():
        s = prefix + (tok,)
        if tok == stop:
            out[s] = p
        else:
            for full, q in enumerate_strings(next_probs, s, stop).items():
                out[full] = p * q
    return out


def local_vs_global(next_probs, valid, stop="$") -> dict:
    """Compare p(y | y valid) with the distribution that per-step masking actually samples from."""
    strings = enumerate_strings(next_probs, stop=stop)
    z = sum(p for s, p in strings.items() if valid(s))
    exact = {s: p / z for s, p in strings.items() if valid(s)}

    def prefix_ok(prefix):  # can this prefix still reach a valid string?
        return any(s[: len(prefix)] == prefix and valid(s) for s in strings)

    def alpha(prefix):  # mass of valid continuations reachable from the prefix
        n = len(prefix)
        tot = sum(p for s, p in strings.items() if s[:n] == prefix)
        return sum(p for s, p in strings.items() if s[:n] == prefix and valid(s)) / tot

    local = {}
    for s in exact:
        prob = 1.0
        for i in range(len(s)):
            dist = next_probs(s[:i])
            allowed = {t: p for t, p in dist.items() if prefix_ok(s[:i] + (t,))}
            prob *= allowed[s[i]] / sum(allowed.values())
        local[s] = prob
    kl = sum(p * math.log(p / local[s]) for s, p in exact.items() if p > 0)
    # Reweight each step by alpha of the extended prefix: this recovers the exact conditional.
    fixed = {}
    for s in exact:
        prob = 1.0
        for i in range(len(s)):
            dist = next_probs(s[:i])
            w = {t: p * alpha(s[:i] + (t,)) for t, p in dist.items()}
            prob *= w[s[i]] / sum(w.values())
        fixed[s] = prob
    return {"exact": exact, "local": local, "alpha_corrected": fixed, "kl_exact_to_local": kl,
            "alpha": {s[:1]: alpha(s[:1]) for s in exact}}
