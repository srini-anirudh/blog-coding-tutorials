"""Section 1: the gap a base model leaves. Figure 1 is schematic; this measures the real thing.

SmolLM2-135M (base) and SmolLM2-135M-Instruct on the same questions. For the base model we try two
inputs: the bare question, and the question inside the chat template it was never trained on.
"""

import argparse
import math

import _common
import torch
from sftlab import data, lm, templates as T

QA = [  # (question, gold substring)
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Egypt?", "Cairo"), ("What is the capital of Canada?", "Ottawa"),
    ("Who wrote Hamlet?", "Shakespeare"), ("What is the chemical symbol for gold?", "Au"),
    ("How many days are in a week?", "seven"), ("What is the largest planet in the solar system?", "Jupiter"),
    ("What gas do plants absorb from the air?", "carbon dioxide"), ("What is the boiling point of water in Celsius?", "100"),
]


def top5(model, tok, ids, dev):
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)
    p = lp.exp()
    top = torch.topk(p, 5)
    ent = float(-(p * lp).sum() / math.log(2))
    return [(tok.decode([i]), round(float(v), 3)) for v, i in zip(top.values, top.indices)], ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="held-out prompts for the rate table")
    ap.add_argument("--max-new", type=int, default=96)
    a = ap.parse_args()
    dev = lm.device()
    out = _common.Report("01_the_gap")
    tok = lm.chat_tokenizer()
    base, ins = lm.load_model(lm.BASE).eval(), lm.load_model(lm.INSTRUCT).eval()
    eot, eos = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    stops = [eot, eos]

    out("1. The next-token distribution right after a question (greedy top 5, probabilities, entropy in bits)")
    for q, gold in QA[:3]:
        raw = tok(q, add_special_tokens=False)["input_ids"]
        chat = T.encode_prompt(tok, [{"role": "user", "content": q}])
        out(f"  Q: {q}")
        for label, m, ids in [("base, bare question", base, raw), ("base, chat template", base, chat),
                              ("instruct, chat template", ins, chat)]:
            t5, h = top5(m, tok, ids, dev)
            out(f"    {label:<24} H={h:5.2f} bits  top5={t5}")
    out("")

    out(f"2. Greedy continuations, {a.max_new} new tokens, first 3 questions")
    for q, gold in QA[:3]:
        raw = tok(q, add_special_tokens=False)["input_ids"]
        chat = T.encode_prompt(tok, [{"role": "user", "content": q}])
        for label, m, ids in [("base, bare", base, raw), ("base, chat", base, chat), ("instruct, chat", ins, chat)]:
            g = lm.greedy(m, tok, [ids], max_new=a.max_new, stop_ids=stops)[0]
            out(f"  [{label}] {g['text']!r}  stopped={g['stopped']}")
        out("")

    # rate table
    alp = data.load_alpaca()
    pool = [e for e in alp if not e["context"] and len(e["prompt"]) < 200 and "?" in e["prompt"]][:100000]
    import random
    random.Random(0).shuffle(pool)
    items = [{"q": q, "gold": g} for q, g in QA] + [{"q": e["prompt"], "gold": None} for e in pool[:a.n]]
    out(f"3. Rates over {len(QA)} factual questions plus {a.n} held-out Alpaca questions, greedy, {a.max_new} new tokens")
    out("   stopped: emitted <|im_end|> or <|endoftext|> within the budget.")
    out("   repeats the question: the question text appears again in the continuation.")
    out("   new question: the continuation contains a '?' that is not part of a repeated question.")
    out("   stop (12): the same, on the 12 short factual questions only (the Alpaca answers are often longer than the budget).")
    out("   gold in first 20 tokens: only for the 12 factual questions.")
    out(f"   {'input':<26} {'stopped':>8} {'stop (12)':>10} {'repeats q':>10} {'new q':>7} {'gold<=20':>9} {'mean new tokens':>16}")
    rows = {}
    for label, m, mode in [("base, bare question", base, "raw"), ("base, chat template", base, "chat"),
                           ("instruct, chat template", ins, "chat")]:
        prompts = [tok(it["q"], add_special_tokens=False)["input_ids"] if mode == "raw"
                   else T.encode_prompt(tok, [{"role": "user", "content": it["q"]}]) for it in items]
        gens = lm.greedy(m, tok, prompts, max_new=a.max_new, stop_ids=stops, batch=50)
        st = rep = newq = gold = 0
        glen = 0
        for it, g in zip(items, gens):
            txt = g["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
            st += g["stopped"]
            r = it["q"].strip() in txt
            rep += r
            rest = txt.replace(it["q"].strip(), "") if r else txt
            newq += "?" in rest
            glen += len(g["ids"])
        for it, g in zip(items[:len(QA)], gens[:len(QA)]):
            gold += it["gold"].lower() in tok.decode(g["ids"][:20]).lower()
        n = len(items)
        st12 = sum(g["stopped"] for g in gens[:len(QA)]) / len(QA)
        out(f"   {label:<26} {st/n:8.2f} {st12:10.2f} {rep/n:10.2f} {newq/n:7.2f} {gold/len(QA):9.2f} {glen/n:16.1f}")
        rows[label] = dict(stopped=st / n, stopped_factual12=st12, repeats_question=rep / n, new_question=newq / n,
                           gold_in_20=gold / len(QA), mean_new_tokens=glen / n)
    out("")
    out("Reading: on a bare question this base model usually does answer (see 'gold<=20'): its top next token is a")
    out("newline, then the answer. What it does not do is stop; it goes on to repeat the question or write a new one.")
    out("Inside a chat template it never saw, it echoes the question and emits junk. So the gap here is turn")
    out("structure and stopping more than knowledge, which qualifies the post's 'followed by another question rather")
    out("than Paris' picture for this model. These are string checks on greedy decodes of a 135M model, not a")
    out("judged evaluation, and one model family.")
    out.save()
    _common.save_json("01_the_gap", {"rates": rows, "n_prompts": len(items)})


if __name__ == "__main__":
    main()
