"""Likelihood displacement and the length confound in a real language model (SmolLM2-135M-Instruct, CPU).

1. Length (Nuance 04): under the frozen model, how much summed log pi(y | x) and the per-token mean fall when a
   correct answer gets longer. The summed form pays for every extra token; the mean pays only for how unlikely
   the extra tokens are on average.
2. Displacement: DPO on 16 hand-written pairs, full fine-tuning, beta = 0.1. Two datasets share the chosen answers:
     near   rejected = the chosen answer with one fact swapped (shares almost every token)
     far    rejected = an unrelated refusal (shares almost nothing)
   We track the change in log pi(y_w) and log pi(y_l) against the reference. A third run adds an NLL term on y_w
   (RPO) to the near dataset.

The reference log-probs are computed once and cached, as the post's practical notes suggest. The sequence
log-prob, batch layout and DPO loss come from C6 (c06-post-training/posttrain).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from prefopt.lm import INSTRUCT_MODEL, c06_dpo_loss, collate_pairs, load_tokenizer, pair_logprobs  # noqa: E402
from prefopt.lm import tokenize_pair  # noqa: E402
from prefopt.report import Report  # noqa: E402

# (question, chosen answer, the same answer with one fact swapped)
DATA = [
    ("What is the capital of France?", "The capital of France is Paris.", "The capital of France is Lyon."),
    ("What is 7 times 8?", "7 times 8 is 56.", "7 times 8 is 54."),
    ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun.",
     "Venus is the planet closest to the Sun."),
    ("How many legs does a spider have?", "A spider has eight legs.", "A spider has six legs."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air.",
     "Plants absorb oxygen from the air."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.",
     "Romeo and Juliet was written by Charles Dickens."),
    ("What is the boiling point of water at sea level in Celsius?",
     "At sea level, water boils at 100 degrees Celsius.", "At sea level, water boils at 90 degrees Celsius."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth.",
     "The Atlantic Ocean is the largest ocean on Earth."),
    ("How many days are in a leap year?", "A leap year has 366 days.", "A leap year has 365 days."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.",
     "The chemical symbol for gold is Ag."),
    ("What is the square root of 81?", "The square root of 81 is 9.", "The square root of 81 is 8."),
    ("Which continent is Kenya in?", "Kenya is in Africa.", "Kenya is in Asia."),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit.",
     "Water freezes at 0 degrees Fahrenheit."),
    ("How many sides does a hexagon have?", "A hexagon has six sides.", "A hexagon has eight sides."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world.",
     "K2 is the tallest mountain in the world."),
    ("What is the main language spoken in Brazil?", "The main language spoken in Brazil is Portuguese.",
     "The main language spoken in Brazil is Spanish."),
]
REFUSAL = "I'm sorry, but I am not able to help with that question right now."
# Longer and shorter correct answers for the length check.
VERBOSE = " This is a well-established fact that you can find in any standard reference book."


def pairs(kind: str):
    out = []
    for q, good, bad in DATA:
        rej = bad if kind == "near" else REFUSAL
        out.append({"prompt": [{"role": "user", "content": q}], "chosen": [{"role": "assistant", "content": good}],
                    "rejected": [{"role": "assistant", "content": rej}]})
    return out


def batch_for(tok, ps):
    return collate_pairs([tokenize_pair(tok, p) for p in ps], pad_id=tok.pad_token_id)


def length_check(model, tok, rep):
    ps = []
    for q, good, _ in DATA:
        ps.append({"prompt": [{"role": "user", "content": q}], "chosen": [{"role": "assistant", "content": good}],
                   "rejected": [{"role": "assistant", "content": good + VERBOSE}]})
    b = batch_for(tok, ps)
    with torch.no_grad():
        s_short, s_long = pair_logprobs(model, b)
        m_short, m_long = pair_logprobs(model, b, average=True)
    n_short, n_long = b["completion_mask"].sum(-1).chunk(2)
    sums, means, lens = torch.cat([s_short, s_long]), torch.cat([m_short, m_long]), torch.cat([n_short, n_long])
    corr = lambda a, c: torch.corrcoef(torch.stack([a.double(), c.double()]))[0, 1].item()  # noqa: E731
    rep.line("1. Length: each correct answer, and the same answer with one true but redundant sentence appended")
    rep.line(f"   mean tokens: short {n_short.double().mean().item():.1f}, long {n_long.double().mean().item():.1f}")
    rep.line(f"   summed log pi: short {s_short.mean().item():.2f}, long {s_long.mean().item():.2f}; "
             f"long is lower in {(s_long < s_short).sum().item()}/16 prompts")
    rep.line(f"   mean log pi per token: short {m_short.mean().item():.3f}, long {m_long.mean().item():.3f}; "
             f"long is lower in {(m_long < m_short).sum().item()}/16 prompts")
    rep.line(f"   correlation with token count over all 32: summed {corr(sums, lens):+.3f}, per-token "
             f"{corr(means, lens):+.3f}")
    return {"sum_short": s_short.mean().item(), "sum_long": s_long.mean().item(), "mean_short": m_short.mean().item(),
            "mean_long": m_long.mean().item(), "corr_sum_len": corr(sums, lens), "corr_mean_len": corr(means, lens)}


def train(kind: str, rpo: float, steps: int, lr: float, beta: float, tok, rep):
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL, dtype=torch.float32)
    b = batch_for(tok, pairs(kind))
    model.eval()  # no dropout in SmolLM2; eval() only to be explicit
    with torch.no_grad():
        ref_w, ref_l = pair_logprobs(model, b)  # cached once
    n_w = b["completion_mask"][:, 1:].sum(-1).chunk(2)[0].double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    logs = []
    for step in range(steps + 1):
        lw, ll = pair_logprobs(model, b)
        loss, info = c06_dpo_loss(lw, ll, ref_w, ref_l, beta)
        if rpo:
            loss = loss + rpo * (-(lw / n_w).mean())
        if step % 5 == 0 or step == steps:
            logs.append({"step": step, "loss": loss.item(), "acc": info["accuracy"].mean().item(),
                         "d_logp_w": (lw - ref_w).mean().item(), "d_logp_l": (ll - ref_l).mean().item(),
                         "frac_w_down": (lw < ref_w).double().mean().item()})
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    name = f"{kind}" + (f" + RPO {rpo}" if rpo else "")
    rep.line(f"\n{name}: beta {beta}, AdamW lr {lr}, full batch of 16 pairs, {steps} steps")
    rep.table(["step", "loss", "pair acc", "change log pi(y_w)", "change log pi(y_l)", "frac y_w below ref"],
              [[x["step"], x["loss"], x["acc"], x["d_logp_w"], x["d_logp_l"], x["frac_w_down"]] for x in logs])
    return {"run": name, "logs": logs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--name", default="08_displacement_lm")
    args = ap.parse_args()
    torch.manual_seed(0)
    rep = Report(args.name)
    rep.line(f"model {INSTRUCT_MODEL}, float32")
    tok = load_tokenizer(INSTRUCT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL, dtype=torch.float32).eval()
    rep.data["length"] = length_check(model, tok, rep)
    del model
    rep.line("\n2. Displacement")
    rep.data["runs"] = [train("near", 0.0, args.steps, args.lr, args.beta, tok, rep),
                        train("far", 0.0, args.steps, args.lr, args.beta, tok, rep),
                        train("near", 1.0, args.steps, args.lr, args.beta, tok, rep)]
    rep.save()


if __name__ == "__main__":
    main()
