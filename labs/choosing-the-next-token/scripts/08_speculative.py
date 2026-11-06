"""Speculative sampling: the accept/reject rule is exact, and greedy speculative output equals target greedy output.

    python labs/choosing-the-next-token/scripts/08_speculative.py

Draft SmolLM2-135M, target SmolLM2-360M (same tokenizer). This script counts target forward passes. It does not
time anything: on CPU, without a tuned serving stack, wall-clock speed-ups would say little about the method.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import lm, speculative  # noqa: E402
from nexttoken.data import Report, prompt_and_reference  # noqa: E402
from nexttoken.dists import softmax  # noqa: E402

r = Report(__file__, "08_speculative")


def tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


r("== Part 1: one token, synthetic p and q over 50 tokens, 200000 draws each ==")
r(f"{'case':<6} {'TV(q, p)':>9} {'TV(spec, p)':>12} {'TV(iid p, p)':>13} {'accept rate':>12} {'1 - TV(q,p)':>12}")
g = torch.Generator().manual_seed(0)
N = 200_000
for case in range(5):
    p = torch.softmax(torch.randn(50, generator=g) * 2, -1)
    q = torch.softmax(torch.log(p) + torch.randn(50, generator=g) * (0.5 + case * 0.5), -1)
    draws = speculative.toy_speculative_draws(p, q, N, g)
    emp = torch.bincount(draws, minlength=50).float() / N
    iid = torch.bincount(torch.multinomial(p, N, replacement=True, generator=g), minlength=50).float() / N
    x = torch.multinomial(q, N, replacement=True, generator=g)
    acc = (torch.rand(N, generator=g) < torch.clamp(p[x] / q[x], max=1)).float().mean()
    r(f"{case:<6} {tv(q, p):9.3f} {tv(emp, p):12.4f} {tv(iid, p):13.4f} {acc:12.3f} {1 - tv(q, p):12.3f}")

tok, draft = lm.load(lm.BASE)
_, target = lm.load(lm.LARGE)
pairs = prompt_and_reference(tok, 6, 32, 64)

r("\n== Part 2: first emitted token, real models, 4000 independent rounds from one prompt, gamma = 4, T = 1 ==")
prompt = pairs[0][0]
g = torch.Generator().manual_seed(1)
first = []
for _ in range(16):
    emitted, _ = speculative.speculative_round(draft, target, prompt.expand(250, -1), 4, 1.0, g)
    first += [e[0] for e in emitted]
p_t = softmax(lm.next_logits(target, prompt))[0]
p_d = softmax(lm.next_logits(draft, prompt))[0]
V = p_t.numel()
emp = torch.bincount(torch.tensor(first), minlength=V).float() / len(first)
iid = torch.bincount(torch.multinomial(p_t, len(first), replacement=True, generator=g), minlength=V).float() / len(first)
r(f"TV(draft p, target p)                    = {tv(p_d, p_t):.4f}")
r(f"TV(speculative first token, target p)    = {tv(emp, p_t):.4f}")
r(f"TV(4000 iid target draws, target p)      = {tv(iid, p_t):.4f}   (pure sampling noise at this sample size)")

r("\n== Part 3: greedy speculative decoding vs target greedy, 6 prompts x 64 tokens, gamma = 4 ==")
same, calls = 0, 0
for prompt, _ in pairs:
    spec, st = speculative.speculative_generate(draft, target, prompt, 64, 4, temperature=0)
    ref = lm.decode(target, prompt, lambda z, h: z.argmax(-1), 64)[0].tolist()
    same += spec == ref
    calls += st["target_calls"]
r(f"identical token sequences: {same} / {len(pairs)}; target forward passes: {calls} (plain greedy: {64 * len(pairs)})")

r("\n== Part 4: sampling at T = 1, 6 prompts x 64 tokens: draft acceptance and tokens per target pass ==")
r(f"{'gamma':<6} {'draft acceptance':>17} {'tokens per target pass':>23} {'target passes':>14}")
for gamma in (1, 2, 4, 6):
    g = torch.Generator().manual_seed(2)
    acc, tpc, n = [], [], 0
    for prompt, _ in pairs:
        _, st = speculative.speculative_generate(draft, target, prompt, 64, gamma, 1.0, g)
        acc.append(st["draft_acceptance"])
        tpc.append(st["tokens_per_target_call"])
        n += st["target_calls"]
    r(f"{gamma:<6} {sum(acc) / len(acc):17.3f} {sum(tpc) / len(tpc):23.2f} {n:14d}")
r("(plain sampling needs one target pass per token: 384 passes for the same 384 tokens)")
r.save()
