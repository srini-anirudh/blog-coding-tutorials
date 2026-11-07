"""Section 5: which tokens receive loss.

A. The assistant-only mask, proved two ways, and the check that masked tokens are still attended to.
B. How many tokens does each masking strategy score? How much of the sequence is boilerplate?
C. Normalisation: per-token against per-example weights.
D. Fine-tune SmolLM2-135M with assistant-only, full-sequence and instruction-modelling losses, in two regimes.
"""

import argparse
import random

import _common
import numpy as np
import torch
from sftlab import checks, data, lm, metrics, pipeline, templates as T
from sftlab.train import SFTConfig, mean_nll


def part_a(out, tok, n):
    out("A. The assistant-only loss, computed three ways, fp32 on CPU, SmolLM2-135M base")
    model = lm.load_model(lm.BASE, dev="cpu").float().eval()
    pool = random.Random(0).sample([e for e in data.load_alpaca() if len(e["prompt"]) + len(e["response"]) < 900], n // 2) \
        + random.Random(0).sample([e for e in data.load_dolly() if len(e["prompt"]) + len(e["response"]) < 900], n // 2)
    d_hf, d_two, rel = [], [], []
    with torch.no_grad():
        for e in pool:
            msgs = data.to_messages(e)
            enc = T.encode_chat(tok, msgs, max_len=400)
            if enc.truncated:
                continue
            a = checks.loss_manual(model, enc)
            b = checks.loss_hf_labels(model, enc)
            c = checks.response_given_prompt(model, tok, msgs)
            d_hf.append(abs(a - b))
            d_two.append(abs(a - c))
            rel.append(abs(a - c) / a)
    out(f"   {len(d_hf)} examples. Sum of NLL over the masked-in tokens (nats):")
    out(f"     (1) manual: logits, per-token NLL, multiply by the mask        (2) Hugging Face loss with labels = -100")
    out(f"     (3) -log p(prompt + response) minus -log p(prompt), no mask at all")
    out(f"     max |(1) - (2)| = {max(d_hf):.2e}      max |(1) - (3)| = {max(d_two):.2e}   (largest relative difference {max(rel):.1e})")
    # masked tokens are attended
    ex = pool[0]
    other = pool[1]
    m1 = data.to_messages(ex)
    m2 = [{"role": "user", "content": other["prompt"]}, m1[1]]
    e1, e2 = T.encode_chat(tok, m1), T.encode_chat(tok, m2)
    with torch.no_grad():
        l1 = checks.nll_per_token(model, e1.ids)
        n_p = len(T.encode_prompt(tok, m1[:1]))
        w1 = torch.tensor(e1.w[1:])
        asst1 = float((l1 * w1).sum() / w1.sum())
        # same reply after a different prompt
        l2 = checks.nll_per_token(model, e2.ids)
        w2 = torch.tensor(e2.w[1:])
        asst2 = float((l2 * w2).sum() / w2.sum())
        # same reply, prompt hidden from attention
        ids = e1.ids
        att = torch.ones(1, len(ids), dtype=torch.long)
        att[0, 1:n_p] = 0  # hide everything except the first token (keeps position 0 as a sink)
        lg = model(input_ids=torch.tensor([ids]), attention_mask=att).logits[0].float()
        l3 = torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(ids[1:]), reduction="none")
        asst3 = float((l3 * w1).sum() / w1.sum())
    out(f"   Masked tokens are still context. One reply, mean assistant-token NLL: after its own prompt {asst1:.3f}; after a different prompt {asst2:.3f};")
    out(f"   with its own prompt hidden from attention {asst3:.3f}. The mask changes which terms are summed; it does not change what the model sees.")
    out("")
    return max(d_hf), max(d_two)


def part_b(out, tok):
    out("B. Tokens scored under each strategy (2,000 random examples per dataset, max 512 tokens)")
    out(f"   {'dataset':<8} {'strategy':<30} {'scored/total':>13} {'mean scored per example':>25}")
    res = {}
    for name, items in [("Alpaca", data.load_alpaca()), ("Dolly", data.load_dolly())]:
        sub = random.Random(0).sample(items, 2000)
        for mode in ["all", "assistant", "weighted"]:
            encs = [T.encode_chat(tok, data.to_messages(e), mode=mode, max_len=512, prompt_weight=0.3) for e in sub]
            tot = sum(len(e) for e in encs)
            sc = sum(sum(e.w[1:]) for e in encs)
            label = {"all": "full sequence", "assistant": "assistant only", "weighted": "instruction modelling (w = 0.3)"}[mode]
            out(f"   {name:<8} {label:<30} {sc / tot:13.3f} {sc / len(encs):25.1f}")
            res[f"{name}_{mode}"] = sc / tot
        enc = [T.encode_chat(tok, data.to_messages(e)) for e in sub]
        sysn = len(tok(T.render(tok, [{'role': 'user', 'content': ''}]).split('<|im_start|>user')[0], add_special_tokens=False)['input_ids'])
        ratio = np.median([(len(e) - e.scored) / max(1, e.scored) for e in enc])
        out(f"   {name:<8} default system prompt = {sysn} tokens, {sysn / np.mean([len(e) for e in enc]):.1%} of an average example; median (non-scored : scored) = {ratio:.2f}")
    out("")
    return res


def part_c(out, tok):
    out("C. Normalisation: how much of a batch's loss weight sits on the longest examples? (Dolly, batches of 16, 500 random batches)")
    items = data.load_dolly()
    encs = [T.encode_chat(tok, data.to_messages(e), max_len=512) for e in random.Random(0).sample(items, 8000)]
    rng = random.Random(1)
    shares_tok, shares_ex = [], []
    for _ in range(500):
        b = rng.sample(encs, 16)
        n = np.array([e.scored for e in b], float)
        order = np.argsort(-n)[:4]  # the 4 longest of 16
        shares_tok.append(n[order].sum() / n.sum())
        shares_ex.append(0.25)
    out(f"   per-token normalisation: the 4 longest of 16 examples carry {np.mean(shares_tok):.1%} of the loss weight on average")
    out(f"   per-example normalisation: they carry 25.0% by construction")
    n_all = np.array([e.scored for e in encs], float)
    out(f"   scored tokens per example: median {np.median(n_all):.0f}, 10th percentile {np.percentile(n_all, 10):.0f}, 90th percentile {np.percentile(n_all, 90):.0f}, max {n_all.max():.0f}")
    out("")


def f1_eval(model, tok, val, max_new=64):
    g = pipeline.completions(model, tok, val, max_new=max_new)
    return float(np.mean([metrics.f1(r["reply"], e["response"]) for r, e in zip(g, val)])), float(np.mean([r["stopped"] for r in g]))


def run_arm(tok, tr, va, mode, norm, cfg, gen_eval):
    enc = pipeline.encode_examples(tok, tr, mode, 384)
    model, _ = pipeline.finetune(tok, enc, SFTConfig(**{**cfg, "norm": norm}))
    val_enc = pipeline.encode_examples(tok, va, "assistant", 384)
    m = mean_nll(model, val_enc, tok.pad_token_id)
    rec = {"token_nll": m["token_nll"], "example_nll": m["example_nll"]}
    if gen_eval:
        rec["f1"], rec["stopped"] = f1_eval(model, tok, va)
    return rec


def part_d(out, tok, seeds, n_val):
    out("D. Fine-tune SmolLM2-135M (base) with different losses. lr 2e-5, batch 16, validation = held-out replies, assistant tokens.")
    out("   NLL columns: mean per scored token, and mean over examples. Mean of seeds, with the std over seeds after '+-'.")
    arms = [("assistant only", "assistant", "token"), ("full sequence", "all", "token"),
            ("instruction modelling w=0.3", "weighted", "token"), ("assistant only, per-example norm", "assistant", "example")]
    results = {}
    # Regime 1: Alpaca, typical
    alp = [e for e in data.load_alpaca() if 10 < len(e["response"]) < 1200]
    # Regime 2: Dolly, long instruction (context), short response
    dol = [e for e in data.load_dolly() if len(e["context"]) > 400 and len(e["response"]) < 200 and e["category"] in ("closed_qa", "information_extraction", "summarization")]
    out(f"   regime 2 pool: {len(dol)} Dolly examples with context > 400 characters and response < 200 characters")
    regimes = [("R1 Alpaca, N=1000, 3 epochs", alp, 1000, 3, False), ("R2a Dolly long-instruction/short-response, N=100, 10 epochs", dol, 100, 10, True),
               ("R2b the same, N=800, 3 epochs", dol, 800, 3, True)]
    for rname, pool, n, ep, gen in regimes:
        out(f"   {rname}")
        hdr = f"     {'loss':<34} {'token NLL':>16} {'example NLL':>16}" + (f" {'token F1':>14} {'stopped':>9}" if gen else "")
        out(hdr)
        base_val = None
        for aname, mode, norm in arms:
            if rname.startswith("R2a") and norm == "example":
                continue
            rs = []
            for seed in seeds:
                tr, va = data.split(pool, n, n_val, seed=seed)
                rs.append(run_arm(tok, tr, va, mode, norm, dict(lr=2e-5, epochs=ep, seed=seed), gen))
                torch.cuda.empty_cache()
            line = f"     {aname:<34} " + "".join(f"{np.mean([r[k] for r in rs]):9.3f}+-{np.std([r[k] for r in rs]):5.3f}" for k in ["token_nll", "example_nll"])
            if gen:
                line += "".join(f"{np.mean([r[k] for r in rs]):9.3f}+-{np.std([r[k] for r in rs]):4.2f}" if k == "f1" else f"{np.mean([r[k] for r in rs]):9.2f}" for k in ["f1", "stopped"])
            out(line)
            results[f"{rname} | {aname}"] = rs
        tr, va = data.split(pool, n, n_val, seed=seeds[0])
        m0 = lm.load_model(lm.BASE)
        zero = mean_nll(m0, pipeline.encode_examples(tok, va, "assistant", 384), tok.pad_token_id)
        out(f"     (no fine-tuning, seed {seeds[0]} validation set: token NLL {zero['token_nll']:.3f}, example NLL {zero['example_nll']:.3f})")
        del m0
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-proof", type=int, default=20)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    out = _common.Report("05_loss_mask")
    tok = lm.chat_tokenizer()
    err = part_a(out, tok, a.n_proof)
    part_b(out, tok)
    part_c(out, tok)
    res = {} if a.skip_train else part_d(out, tok, a.seeds, a.n_val)
    out.save()
    _common.save_json("05_loss_mask", {"proof_max_abs_diff": err, "arms": res})


if __name__ == "__main__":
    main()
