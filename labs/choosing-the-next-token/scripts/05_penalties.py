"""Degeneration controls: the sign trap, the shift problem, loop breaking under greedy, and the cost to code.

    python labs/choosing-the-next-token/scripts/05_penalties.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import lm, penalties as pen  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402
from nexttoken.metrics import repeated_ngram_fraction  # noqa: E402

r = Report(__file__, "05_penalties")

r("== The sign trap (synthetic logits; token 1 was already generated, its logit is negative) ==")
z = torch.tensor([[2.0, -1.0, -3.0, 0.5]])
h = torch.tensor([[1]])
for name, f in [("no penalty", lambda x: x), ("naive z / 1.5", lambda x: pen.repetition_penalty_naive(x, h, 1.5)),
                ("CTRL branch", lambda x: pen.repetition_penalty(x, h, 1.5))]:
    r(f"{name:<15} logit of token 1 {f(z)[0, 1]:6.2f}   p(token 1) = {torch.softmax(f(z), -1)[0, 1]:.4f}")

r("\n== Shift invariance (same logits, plus a constant 5.0 before the penalty; tokens 0 and 2 seen) ==")
z = torch.tensor([[2.0, 1.0, 0.5, -0.5]])
h = torch.tensor([[0, 2]])
for name, f in [("repetition 1.5", lambda x: pen.repetition_penalty(x, h, 1.5)),
                ("frequency 0.7", lambda x: pen.frequency_penalty(x, h, 0.7))]:
    a, b = torch.softmax(f(z), -1)[0], torch.softmax(f(z + 5.0), -1)[0]
    r(f"{name:<15} p before shift {[round(v, 3) for v in a.tolist()]}  after {[round(v, 3) for v in b.tolist()]}")

tok, model = lm.load(lm.BASE)
PROCESSORS = [
    ("none", lambda z, h: z),
    ("repetition 1.2 (CTRL)", lambda z, h: pen.repetition_penalty(z, h, 1.2)),
    ("repetition 1.2 naive", lambda z, h: pen.repetition_penalty_naive(z, h, 1.2)),
    ("frequency 0.5", lambda z, h: pen.frequency_penalty(z, h, 0.5)),
    ("presence 0.5", lambda z, h: pen.presence_penalty(z, h, 0.5)),
    ("no-repeat 3-gram", lambda z, h: pen.no_repeat_ngram(z, h, 3)),
]

pairs = prompt_and_reference(tok, 10, 32, 100)
r("\n== Greedy decoding, SmolLM2-135M, 10 WikiText-2 prompts, 100 new tokens. Penalties use prompt + output ==")
r(f"{'processor':<24} {'mean rep-4':>10} {'prompts with rep-4 > 0.3':>25}")
for name, proc in PROCESSORS:
    reps = []
    for prompt, _ in pairs:
        gen = lm.decode(model, prompt, lambda z, h: proc(z, h).argmax(-1), 100)
        reps.append(repeated_ngram_fraction(gen[0].tolist()))
    r(f"{name:<24} {sum(reps) / len(reps):10.3f} {sum(x > 0.3 for x in reps):>25}")

CODE_PROMPTS = [
    ("def mean(values):\n    total = 0\n    for value in values:\n", ["total", "values", "value"]),
    ("def count_words(text):\n    counts = {}\n    for word in text.split():\n", ["counts", "word"]),
    ("class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n", ["self.items", "item"]),
]
r("\n== Good repetition: greedy code completion, 60 new tokens. Count of times the identifiers reappear ==")
r(f"{'processor':<24} " + " ".join(f"{'prompt ' + str(i + 1):>9}" for i in range(len(CODE_PROMPTS))))
outputs = {}
for name, proc in PROCESSORS:
    counts = []
    for i, (code, names) in enumerate(CODE_PROMPTS):
        ids = tok(code, return_tensors="pt").input_ids
        text = tok.decode(lm.decode(model, ids, lambda z, h: proc(z, h).argmax(-1), 60)[0])
        counts.append(sum(text.count(n) for n in names))
        if i == 0:
            outputs[name] = text
    r(f"{name:<24} " + " ".join(f"{c:9d}" for c in counts))
r(f"\n== Completions of code prompt 1 ==\n{CODE_PROMPTS[0][0]!r}")
for name, text in outputs.items():
    r(f"\n[{name}]\n{text[:300]!r}")
r.save()
