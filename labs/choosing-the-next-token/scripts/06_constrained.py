"""Constrained decoding: a per-state token mask over the real vocabulary, and why masking is not conditioning.

    python labs/choosing-the-next-token/scripts/06_constrained.py

Part 1 asks SmolLM2-135M-Instruct to turn a sentence about a person into {"name": "...", "age": N}, with and
without the automaton mask. Part 2 computes, exactly, how far step-wise masking is from p(y | y in L) on small random
autoregressive models, and how importance resampling closes the gap.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import constrained as cd, lm  # noqa: E402
from nexttoken.data import Report  # noqa: E402

r = Report(__file__, "06_constrained")
tok, model = lm.load(lm.INSTRUCT)
eos = tok.convert_tokens_to_ids("<|im_end|>")
V = model.config.vocab_size
dfa = cd.CharDFA(cd.PERSON_PATTERN)

t0 = time.perf_counter()
strings = [tok.decode([i], clean_up_tokenization_spaces=False) if i < len(tok) else "" for i in range(V)]
for i in tok.all_special_ids:
    strings[i] = ""
index = cd.TokenIndex(dfa, strings, eos)
r(f"== Part 1: JSON person records, {lm.INSTRUCT} ==")
r(f"vocabulary strings decoded in {time.perf_counter() - t0:.1f} s")

PEOPLE = [("Maria Lopez", 34), ("John Carter", 52), ("Aiko Tanaka", 27), ("Peter Novak", 41), ("Fatima Khan", 19),
          ("Lucas Silva", 63), ("Emma Brown", 8), ("Omar Haddad", 45), ("Sofia Rossi", 30), ("David Kim", 71),
          ("Grace Obi", 23), ("Henrik Berg", 56)]
TEMPLATES = ["{name} is {age} years old.", "At {age}, {name} still runs every morning.",
             "{name}, aged {age}, works as a nurse."]


def prompt_ids(name, age, j):
    sentence = TEMPLATES[j % len(TEMPLATES)].format(name=name, age=age)
    msg = (f'Extract the person as JSON with keys "name" and "age". Reply with only the JSON object.\n'
           f"Text: {sentence}")
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"]


def constrained_pick(temperature, g):
    state = {"s": dfa.start}

    def pick(logits, _h):
        mask = index.mask(state["s"], V)
        z = logits.masked_fill(~mask, float("-inf"))
        t = z.argmax(-1) if temperature == 0 else torch.multinomial(torch.softmax(z / temperature, -1), 1, generator=g)[:, 0]
        state["s"] = index.allowed(state["s"])[int(t)]
        return t

    return pick


def check(text, name, age):
    """(matches the exact pattern, reply contains parseable JSON with both keys, name and age both right)."""
    text = text.split("<|im_end|>")[0].strip()
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    parsed = isinstance(obj, dict) and "name" in obj and "age" in obj
    correct = parsed and obj["name"] == name and str(obj["age"]) == str(age)
    return dfa.matches(text), parsed, bool(correct)


rows = {}
for mode in ["unconstrained greedy", "constrained greedy", "unconstrained T=1", "constrained T=1"]:
    g = torch.Generator().manual_seed(0)
    T = 0 if "greedy" in mode else 1.0
    res, example = [], None
    for j, (name, age) in enumerate(PEOPLE):
        ids = prompt_ids(name, age, j)
        if mode.startswith("constrained"):
            gen = lm.decode(model, ids, constrained_pick(T, g), 48, eos_id=eos)
        else:
            pick = (lambda z, h: z.argmax(-1)) if T == 0 else (
                lambda z, h: torch.multinomial(torch.softmax(z, -1), 1, generator=g)[:, 0])
            gen = lm.decode(model, ids, pick, 48, eos_id=eos)
        text = tok.decode(gen[0])
        res.append(check(text, name, age))
        example = example or text.split("<|im_end|>")[0]
    rows[mode] = res
    r(f"\n[{mode}] example: {example[:120]!r}")
r(f"\nautomaton states visited and indexed: {len(index.cache)}; "
  f"mean allowed tokens per state: {sum(len(v) for v in index.cache.values()) / len(index.cache):.0f} of {V}")
r(f"\n{'mode':<22} {'exact match of pattern':>23} {'parseable JSON in reply':>23} {'name and age right':>19}")
for mode, res in rows.items():
    n = len(res)
    r(f"{mode:<22} {sum(a for a, _, _ in res):>17d} / {n} {sum(b for _, b, _ in res):>17d} / {n} "
      f"{sum(c for _, _, c in res):>13d} / {n}")
r("(the first constrained call per state builds its mask; later calls are dictionary lookups)")

r("\n== Part 2: masking vs conditioning, exact enumeration (synthetic models, vocabulary 3, length 5) ==")
r("L = sequences that end in token 2 and never contain two 0s in a row.")
r("TV(masked, exact) is the distortion. Importance resampling: draw K particles from the masked sampler,")
r("weight each by p(y) / q(y), keep one in proportion to its weight; 20000 rounds.")


def in_lang(y):
    return y[-1] == 2 and all(not (a == 0 and b == 0) for a, b in zip(y, y[1:]))


r(f"{'seed':<6} {'TV masked':>10} " + " ".join(f"{'K=' + str(k):>8}" for k in (1, 4, 16, 64)))
for seed in range(5):
    table = cd.toy_lm(3, 5, seed=seed)
    exact = cd.exact_conditional(table, 3, 5, in_lang)
    tv = cd.total_variation(cd.masked_distribution(table, 3, 5, in_lang), exact)
    fixed = [cd.total_variation(cd.importance_resample(table, 3, 5, in_lang, k, 20000, seed), exact)
             for k in (1, 4, 16, 64)]
    r(f"{seed:<6} {tv:10.3f} " + " ".join(f"{f:8.3f}" for f in fixed))
r("(K=1 is the masked sampler itself; the floor near 0.01-0.02 is the sampling noise of 20000 draws)")
r.save()
