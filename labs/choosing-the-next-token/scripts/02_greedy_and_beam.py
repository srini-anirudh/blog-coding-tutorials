"""Greedy myopia on the post's two-step tree, then beam search and its length bias on a real instruct model.

    python labs/choosing-the-next-token/scripts/02_greedy_and_beam.py

The real-model part translates short English sentences to French with SmolLM2-135M-Instruct. For each sentence it
compares greedy, beam k=4 and beam k=8, with and without GNMT length normalization, against the score of the empty
output (emitting end-of-turn immediately). It does not run exact search, so Stahlberg and Byrne's result is not
reproduced. It only checks whether the empty output already beats what beam search returns.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexttoken import lm, search  # noqa: E402
from nexttoken.data import Report  # noqa: E402

r = Report(__file__, "02_greedy_and_beam")

r("== The post's two-step tree ==")
tree = search.post_example_tree()
show = lambda h: " ".join(tree.vocab[i] for i in h.tokens)  # noqa: E731
g = search.greedy(tree, tree.eos_id, 5)
r(f"greedy        : {show(g):<12} p = {math.exp(g.logprob):.3f}")
for k in (1, 2):
    b = search.beam_search(tree, tree.eos_id, k, 5)[0]
    r(f"beam k={k}      : {show(b):<12} p = {math.exp(b.logprob):.3f}")
for h in search.exhaustive(tree, 5)[:3]:
    r(f"exhaustive    : {show(h):<12} p = {math.exp(h.logprob):.3f}")

SENTENCES = [
    "The cat is sleeping on the chair.",
    "I would like a cup of coffee, please.",
    "The train leaves at eight o'clock tomorrow morning.",
    "My brother lives in a small house near the river.",
    "We are going to the museum on Saturday with our friends.",
    "She has been reading the same book for three weeks.",
    "The weather was cold, so we stayed inside and played cards.",
    "Can you tell me where the nearest pharmacy is?",
]

tok, model = lm.load(lm.INSTRUCT)
eos = tok.convert_tokens_to_ids("<|im_end|>")
r(f"\n== Translation, {lm.INSTRUCT}, end token <|im_end|>, max 40 new tokens ==")
configs = [("greedy", 1, 0.0), ("beam k=4", 4, 0.0), ("beam k=8", 8, 0.0), ("beam k=4 a=1", 4, 1.0),
           ("beam k=8 a=1", 8, 1.0)]
tally = {name: {"len": [], "lp": [], "empty_wins": 0} for name, _, _ in configs}
for s in SENTENCES:
    msgs = [{"role": "user", "content": f"Translate to French: {s}"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    empty = float(torch.log_softmax(lm.next_logits(model, prompt), -1)[0, eos])
    r(f"\n{s}\n  empty output: log p = {empty:.2f}")
    for name, k, alpha in configs:
        h = search.beam_search(search.HFStepper(model, prompt), eos, k, 40, alpha)[0]
        text = tok.decode([t for t in h.tokens if t != eos]).strip().replace("\n", " ")
        n = len(h.tokens)
        tally[name]["len"].append(n)
        tally[name]["lp"].append(h.logprob)
        tally[name]["empty_wins"] += empty > h.logprob
        r(f"  {name:<13} len {n:2d}  log p {h.logprob:7.2f}  {'done' if h.finished else 'cut '}  {text[:90]}")

r("\n== Summary over the 8 sentences ==")
r("decoder        mean length  mean log p   sentences where the empty output has higher log p than the result")
for name, t in tally.items():
    r(f"{name:<14} {sum(t['len']) / 8:10.1f} {sum(t['lp']) / 8:11.2f}   {t['empty_wins']} / 8")
r.save()
