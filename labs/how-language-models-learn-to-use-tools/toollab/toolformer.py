"""Toolformer's filter: keep a call only if its result makes the next few tokens easier to predict.

    L_i(z)  = - sum_{j >= i} w_{j-i} log p(x_j | z, x_{1:j-1})
    L_plus  = L_i(e(c, r))                       call and result as the prefix z
    L_minus = min(L_i(empty), L_i(e(c, empty)))  no call, or the call with its result withheld
    keep the call  iff  L_minus - L_plus >= tau_f

As in the paper, the call is given to the model as a prefix z rather than spliced into the text,
because a model that has never seen calls reads a call in mid-sentence as noise.
"""

from __future__ import annotations

import random

import torch

ARROW = "→"


def toolformer_weights(span: int = 5) -> torch.Tensor:
    """w_t = max(0, 1 - 0.2 t), normalised. Nonzero for t = 0..4, so only five tokens get credit."""
    raw = torch.clamp(1 - 0.2 * torch.arange(span + 1, dtype=torch.float64), min=0)
    raw = raw[raw > 0]
    return raw / raw.sum()


def encode_call(call: str, result: str | None) -> str:
    """e(c, r) as text. With result None this is e(c, empty): the call with its answer withheld."""
    return f"[{call} {ARROW} {result}]" if result is not None else f"[{call}]"


@torch.no_grad()
def weighted_loss(model, tok, before: str, after: str, prefix: str | None = None) -> float:
    """L_i(z) for the call site between `before` and `after`. `prefix` is z (None means the empty prefix)."""
    w = toolformer_weights()
    head = tok(prefix + "\n", add_special_tokens=False)["input_ids"] if prefix else []
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ctx = bos + head + tok(before, add_special_tokens=False)["input_ids"]
    tgt = tok(after, add_special_tokens=False)["input_ids"][: len(w)]
    ids = torch.tensor([ctx + tgt])
    logp = torch.log_softmax(model(ids).logits[0].double(), -1)
    n = len(ctx)
    nll = torch.stack([-logp[n - 1 + j, t] for j, t in enumerate(tgt)])
    ww = w[: len(tgt)] / w[: len(tgt)].sum()
    return float((ww * nll).sum())


def filter_call(model, tok, before: str, after: str, call: str, result: str) -> dict:
    l_empty = weighted_loss(model, tok, before, after, None)
    l_gesture = weighted_loss(model, tok, before, after, encode_call(call, None))
    l_plus = weighted_loss(model, tok, before, after, encode_call(call, result))
    l_minus = min(l_empty, l_gesture)
    return {"L_empty": l_empty, "L_call_no_result": l_gesture, "L_plus": l_plus, "L_minus": l_minus,
            "gain": l_minus - l_plus, "gain_without_min": l_empty - l_plus}


# ---- A small corpus with three kinds of candidate call per call site ----

CAPITALS = [("Kenya", "Nairobi"), ("Mongolia", "Ulaanbaatar"), ("Ghana", "Accra"), ("Laos", "Vientiane"),
            ("Uzbekistan", "Tashkent"), ("Senegal", "Dakar"), ("Latvia", "Riga"), ("Nepal", "Kathmandu"),
            ("Ecuador", "Quito"), ("Bhutan", "Thimphu"), ("France", "Paris"), ("Japan", "Tokyo")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def corpus(seed: int = 0) -> list[dict]:
    """Call sites with a useful call, a call whose text already gives the answer away, and a useless one."""
    rng = random.Random(seed)
    sites = []
    for i, (country, capital) in enumerate(CAPITALS):
        other = CAPITALS[(i + 5) % len(CAPITALS)]
        sites.append({"tool": "QA", "before": f"The capital of {country} is", "after": f" {capital}.",
                      "calls": {"useful": (f'QA("What is the capital of {country}?")', capital),
                                "answer_in_call": (f'QA("Is {capital} the capital of {country}?")', "Yes"),
                                "irrelevant": (f'QA("What is the capital of {other[0]}?")', other[1])}})
    for _ in range(12):
        b = rng.randint(200, 2000)
        a = rng.randint(20, b - 20)
        r = f"{a / b:.2f}"
        sites.append({"tool": "Calculator", "before": f"Of the {b} applicants, {a} were accepted, a rate of",
                      "after": f" {r}.",
                      "calls": {"useful": (f"Calculator({a} / {b})", r),
                                "answer_in_call": (f"Calculator({r} * 1)", r),
                                "irrelevant": (f"Calculator({a} + {b})", str(a + b))}})
    for _ in range(8):
        d, m, day = rng.choice(DAYS), rng.choice(MONTHS), rng.randint(1, 28)
        sites.append({"tool": "Calendar", "before": "Today is", "after": f" {d}, {m} {day}, 2026.",
                      "calls": {"useful": ("Calendar()", f"Today is {d}, {m} {day}, 2026."),
                                "answer_in_call": (f'Calendar("{d}, {m} {day}")', f"{m} {day}"),
                                "irrelevant": ("Calculator(7 + 1)", "8")}})
    return sites
