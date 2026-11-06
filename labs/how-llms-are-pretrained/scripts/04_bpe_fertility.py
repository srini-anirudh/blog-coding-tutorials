"""Tokenization section and Figure 07: BPE from scratch, the merge-rank truncation fact, and fertility.

1. Train byte-level BPE on WikiText-2 with our code, at two sizes, and check that the small
   tokenizer equals the large one cut to its first merges.
2. Repeat the check with the Hugging Face `tokenizers` BPE trainer.
3. Bytes per token for our tokenizers and three public ones, on English, Python and six languages.
4. What fertility costs: the 6ND compute multiplier and the fp32 logits tensor.
"""

import argparse
import time

import _common
from ptlab import bpe, corpora

SENTENCES = {  # the same sentence, translated for this lab
    "English": "The library opens at nine in the morning and closes at six in the evening.",
    "French": "La bibliothèque ouvre à neuf heures du matin et ferme à six heures du soir.",
    "German": "Die Bibliothek öffnet um neun Uhr morgens und schließt um sechs Uhr abends.",
    "Hindi": "पुस्तकालय सुबह नौ बजे खुलता है और शाम छह बजे बंद होता है।",
    "Telugu": "గ్రంథాలయం ఉదయం తొమ్మిది గంటలకు తెరుచుకుంటుంది మరియు సాయంత్రం ఆరు గంటలకు మూసివేయబడుతుంది.",
    "Chinese": "图书馆早上九点开门，晚上六点关门。",
}
PUBLIC = {"GPT-2": "openai-community/gpt2", "SmolLM2": "HuggingFaceTB/SmolLM2-135M",
          "Qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct"}


def hf_bpe_merges(texts, vocab_size):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False, special_tokens=[])
    tok.train_from_iterator(texts, trainer)
    import json
    return [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bytes", type=int, default=2_000_000)
    ap.add_argument("--small", type=int, default=500)
    ap.add_argument("--large", type=int, default=2000)
    args = ap.parse_args()
    out = _common.Report("04_bpe_fertility")
    docs, n = [], 0
    for d in corpora.wikitext_docs("train"):
        docs.append(d)
        n += len(d.encode())
        if n >= args.train_bytes:
            break
    out(f"BPE training text: {len(docs)} WikiText-2 articles, {n:,} bytes\n")

    t0 = time.time()
    large = bpe.train_bpe(docs, args.large)
    t1 = time.time()
    small = bpe.train_bpe(docs, args.small)
    out(f"our BPE: {args.large} merges in {t1 - t0:.1f} s, {args.small} merges in {time.time() - t1:.1f} s")
    same = large[: args.small] == small
    out(f"first {args.small} merges of the {args.large}-merge run == the {args.small}-merge run: {same}")
    tl, ts = bpe.BPETokenizer(large).truncated(args.small), bpe.BPETokenizer(small)
    val = "\n".join(corpora.wikitext_docs("validation")[:20])
    out(f"encodings of {len(val):,} validation characters identical: {tl.encode(val) == ts.encode(val)}")
    big_tok = bpe.BPETokenizer(large)
    assert big_tok.decode(big_tok.encode(val)) == val
    out("round trip decode(encode(x)) == x: True")
    out(f"first 10 merges: {[big_tok.vocab[256 + i].decode('utf-8', 'replace') for i in range(10)]}")

    hl, hs = hf_bpe_merges(docs, 256 + args.large), hf_bpe_merges(docs, 256 + args.small)
    k = min(len(hs), len(hl))
    first_diff = next((i for i in range(k) if hl[i] != hs[i]), None)
    out(f"\nHugging Face tokenizers trainer: {len(hl)} and {len(hs)} merges learned; "
        f"small run is a prefix of the large run: {hl[: len(hs)] == hs}"
        + ("" if first_diff is None else f" (first difference at merge {first_diff})"))

    from transformers import AutoTokenizer

    toks = {f"ours, {args.small} merges": ts.encode, f"ours, {args.large} merges": big_tok.encode}
    sizes = {f"ours, {args.small} merges": 256 + args.small, f"ours, {args.large} merges": 256 + args.large}
    for name, repo in PUBLIC.items():
        t = AutoTokenizer.from_pretrained(repo)
        toks[name] = lambda s, t=t: t(s, add_special_tokens=False)["input_ids"]
        sizes[name] = len(t)
    code = "\n".join(corpora.stdlib_code_docs("val")[:3])
    texts = {"English prose": val, "Python code": code}
    out("\nBytes per token (higher compresses better)")
    out(f"{'tokenizer':<20} {'vocab':>8} " + "".join(f"{k:>15}" for k in texts))
    bpt = {}
    for name, enc in toks.items():
        bpt[name] = {k: bpe.bytes_per_token(len(v.encode()), len(enc(v))) for k, v in texts.items()}
        out(f"{name:<20} {sizes[name]:>8,} " + "".join(f"{bpt[name][k]:>15.2f}" for k in texts))

    out("\nOne sentence, six languages: tokens, and tokens relative to English")
    out(f"{'tokenizer':<20} " + "".join(f"{lang:>13}" for lang in SENTENCES))
    for name, enc in toks.items():
        counts = {lang: len(enc(s)) for lang, s in SENTENCES.items()}
        out(f"{name:<20} " + "".join(f"{counts[lang]:>6} ({counts[lang] / counts['English']:3.1f}x)" for lang in SENTENCES))
    out("UTF-8 bytes:         " + "".join(f"{len(s.encode()):>13}" for s in SENTENCES.values()))

    out("\nDigits: how '2024 2025 1234567' is split")
    for name in ["GPT-2", "SmolLM2", "Qwen2.5"]:
        t = AutoTokenizer.from_pretrained(PUBLIC[name])
        out(f"  {name:<8} {[t.decode([i]) for i in t('2024 2025 1234567', add_special_tokens=False)['input_ids']]}")

    out("\nWhat fertility costs. C = 6 N D with D = bytes x tokens per byte.")
    worst = min(bpt, key=lambda k: bpt[k]["Python code"])
    best = max(bpt, key=lambda k: bpt[k]["Python code"])
    ratio = bpt[best]["Python code"] / bpt[worst]["Python code"]
    out(f"  On Python code, {worst} needs {ratio:.2f}x the tokens of {best}, so {ratio:.2f}x the training FLOPs for the same bytes.")
    gb = bpe.logits_bytes(8, 8192, 100_000) / 1e9
    out(f"  fp32 logits for batch 8 x 8192 tokens x 100,000 vocab: {gb:.1f} GB for one tensor.")
    out.save()


if __name__ == "__main__":
    main()
