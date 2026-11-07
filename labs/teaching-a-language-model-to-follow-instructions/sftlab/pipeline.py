"""Glue used by several scripts: encode example lists, fine-tune a fresh base model, generate and score."""

from __future__ import annotations

import copy

import torch

from . import data, lm, templates as T
from .train import SFTConfig, train


def encode_examples(tok, exs: list[dict], mode: str = "assistant", max_len: int = 384, prompt_weight: float = 0.3,
                    system: str | None = None) -> list[T.Encoded]:
    out = []
    for e in exs:
        msgs = data.to_messages(e)
        if system is not None:
            msgs = [{"role": "system", "content": system}] + msgs
        out.append(T.encode_chat(tok, msgs, mode=mode, max_len=max_len, prompt_weight=prompt_weight))
    return out


def prompt_ids(tok, exs: list[dict], system: str | None = None) -> list[list[int]]:
    res = []
    for e in exs:
        msgs = [{"role": "user", "content": e["prompt"]}]
        if system is not None:
            msgs = [{"role": "system", "content": system}] + msgs
        res.append(T.encode_prompt(tok, msgs))
    return res


_CACHE: dict = {}


def fresh_model(repo: str = lm.BASE, dev: str | None = None):
    """A new copy of the pretrained weights (loaded once per process, then deep-copied)."""
    if repo not in _CACHE:
        _CACHE[repo] = lm.load_model(repo, dev="cpu")
    return copy.deepcopy(_CACHE[repo]).to(dev or lm.device())


def finetune(tok, train_enc, cfg: SFTConfig, repo: str = lm.BASE, on_epoch=None):
    model = fresh_model(repo)
    log = train(model, train_enc, tok.pad_token_id, cfg, on_epoch=on_epoch)
    return model, log


def completions(model, tok, exs, max_new=128, system=None, batch=64, **kw) -> list[dict]:
    """Generate the assistant reply for each example's prompt. 'text' is the reply with the turn-end token removed."""
    g = lm.greedy(model, tok, prompt_ids(tok, exs, system), max_new=max_new, batch=batch, **kw)
    for r in g:
        r["reply"] = r["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return g


def wiki_encoded(tok, n: int = 100, max_len: int = 256) -> list[T.Encoded]:
    """Web-text paragraphs (WikiText-2 validation), every token scored: a probe for what fine-tuning costs elsewhere."""
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    paras = [t.strip() for t in wt["text"] if len(t.split()) > 60][:n]
    out = []
    for p in paras:
        ids = tok(p, add_special_tokens=False)["input_ids"][:max_len]
        out.append(T.Encoded(ids, [1.0] * len(ids)))
    return out


def behaviour_eval(model, tok, val_enc, val_ex, wiki, train_resps=None, n_gen=100, n_div=20, k_div=8, max_new=128, seed=0):
    """One bundle of measurements used by the 'how much SFT' script."""
    from . import metrics
    from .train import mean_nll
    r = {}
    m = mean_nll(model, val_enc, tok.pad_token_id)
    r["val_nll"] = m["token_nll"]
    r["wiki_nll"] = mean_nll(model, wiki, tok.pad_token_id)["token_nll"]
    g = completions(model, tok, val_ex[:n_gen], max_new=max_new)
    r["stop"] = sum(x["stopped"] for x in g) / len(g)
    r["len"] = sum(len(x["ids"]) for x in g) / len(g)
    if train_resps is not None:
        tw = [set(metrics.words(t)) for t in train_resps]
        near = 0
        exact = 0
        for x in g:
            rw = set(metrics.words(x["reply"]))
            if not rw:
                continue
            best = max(len(rw & t) / max(1, len(rw | t)) for t in tw)
            near += best >= 0.8
            exact += x["reply"].strip() in train_resps
        r["regurgitate"] = near / len(g)
    prompts = [p for p in prompt_ids(tok, val_ex[:n_div]) for _ in range(k_div)]
    s = lm.greedy(model, tok, prompts, max_new=max_new, batch=160, sample=True, seed=seed)
    d = []
    for i in range(n_div):
        reps = [x["text"].replace("<|im_end|>", "") for x in s[i * k_div:(i + 1) * k_div]]
        d.append(metrics.distinct_n(reps, 2))
    r["distinct2"] = sum(d) / len(d)
    return r


SYS_DEFAULT = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def serving_variants(p: str) -> dict[str, str]:
    """The same user prompt rendered with the right ChatML template and with five small mistakes."""
    s, e = "<|im_start|>", "<|im_end|>"
    return {
        "correct template": f"{s}system\n{SYS_DEFAULT}{e}\n{s}user\n{p}{e}\n{s}assistant\n",
        "no newline after 'assistant'": f"{s}system\n{SYS_DEFAULT}{e}\n{s}user\n{p}{e}\n{s}assistant",
        "extra blank line after header": f"{s}system\n{SYS_DEFAULT}{e}\n{s}user\n{p}{e}\n{s}assistant\n\n",
        "no system turn": f"{s}user\n{p}{e}\n{s}assistant\n",
        "no newline after user turn": f"{s}system\n{SYS_DEFAULT}{e}\n{s}user\n{p}{e}{s}assistant\n",
        "plain 'User:/Assistant:' text": f"User: {p}\nAssistant:",
    }


@torch.no_grad()
def serve_eval(model, tok, val: list[dict], max_new: int = 96) -> dict[str, dict]:
    """For each serving variant: NLL per token of the reference reply given that prompt, share of greedy replies that
    ended with the turn-end token, mean new tokens, and the share whose first 30 characters differ from the correct template's."""
    import numpy as np
    names = list(serving_variants("x"))
    res, ref_text = {}, None
    for name in names:
        prompts = [tok(serving_variants(e["prompt"])[name], add_special_tokens=False)["input_ids"] for e in val]
        nll = tot = 0.0
        for e, p in zip(val, prompts):
            r = tok(e["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            lg = model(input_ids=torch.tensor([p + r], device=model.device)).logits[0, len(p) - 1:-1].float()
            nll += float(torch.nn.functional.cross_entropy(lg, torch.tensor(r, device=model.device), reduction="sum"))
            tot += len(r)
        g = lm.greedy(model, tok, prompts, max_new=max_new, batch=50)
        texts = [x["text"] for x in g]
        if ref_text is None:
            ref_text = texts
        res[name] = dict(nll=nll / tot, stopped=float(np.mean([x["stopped"] for x in g])),
                         differs=float(np.mean([a[:30] != b[:30] for a, b in zip(texts, ref_text)])),
                         mean_new=float(np.mean([len(x["ids"]) for x in g])))
    return res


def print_serve(out, res):
    out(f"   {'serving format':<34} {'ref. reply NLL/token':>21} {'stopped':>8} {'reply starts differently':>26} {'mean new tokens':>16}")
    for name, r in res.items():
        out(f"   {name:<34} {r['nll']:21.3f} {r['stopped']:8.2f} {r['differs']:26.2f} {r['mean_new']:16.1f}")
