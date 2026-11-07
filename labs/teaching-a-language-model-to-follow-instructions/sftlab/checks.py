"""Exact checks used by scripts and tests: two ways to compute the masked loss, exploded multi-turn
examples, gradient comparison, packed rows with and without a document mask."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import templates as T


def _logits(model, ids, att=None, **kw):
    return model(input_ids=torch.tensor([ids]), attention_mask=att, **kw).logits[0].float()


def nll_per_token(model, ids: list[int]) -> torch.Tensor:
    """(L-1,) NLL of every token after the first, fp32, no padding."""
    lg = _logits(model, ids)
    return F.cross_entropy(lg[:-1], torch.tensor(ids[1:]), reduction="none")


def loss_manual(model, enc: T.Encoded) -> float:
    """Sum of NLL over scored tokens, from the logits."""
    nll = nll_per_token(model, enc.ids)
    w = torch.tensor(enc.w[1:])
    return float((nll * w).sum().detach())


def loss_hf_labels(model, enc: T.Encoded) -> float:
    """Hugging Face's own loss with labels = -100 on masked tokens, converted back from a mean to a sum."""
    labels = torch.tensor([[i if wt > 0 else -100 for i, wt in zip(enc.ids, enc.w)]])
    out = model(input_ids=torch.tensor([enc.ids]), labels=labels)
    n = int((labels[0, 1:] != -100).sum())
    return float(out.loss) * n


def response_given_prompt(model, tok, messages) -> float:
    """-log p(response | prompt) computed without any mask: score prompt + response, then subtract the score of the
    prompt alone. Valid because causal attention makes the prompt tokens' NLL independent of what follows."""
    prompt_ids = T.encode_prompt(tok, messages[:-1])
    enc = T.encode_chat(tok, messages, mode="assistant")
    assert enc.ids[:len(prompt_ids)] == prompt_ids, "prompt is not a prefix of the full sequence"
    last = max(i for i, w in enumerate(enc.w) if w > 0)       # the template's trailing newline after the turn end is not part of the response
    full = float(nll_per_token(model, enc.ids[:last + 1]).sum())
    prompt_only = float(nll_per_token(model, prompt_ids).sum())
    return full - prompt_only


def explode(tok, messages) -> list[T.Encoded]:
    """One example per assistant turn: the conversation cut after that turn, only that turn scored."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            out.append(T.encode_chat(tok, messages[:i + 1], mode="last"))
    return out


def grad_of_sum_loss(model, encs: list[T.Encoded]) -> torch.Tensor:
    """Flattened gradient of the SUM over examples of the summed weighted NLL."""
    model.zero_grad(set_to_none=True)
    for e in encs:
        nll = nll_per_token(model, e.ids)
        (nll * torch.tensor(e.w[1:])).sum().backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    return g


def packed_nll(model, row_ids: list[list[int]], doc_mask: bool, reset_pos: bool) -> list[torch.Tensor]:
    """Pack documents into one row. Returns, per document, the NLL of its tokens 2..n. Needs eager attention."""
    ids = [t for d in row_ids for t in d]
    L = len(ids)
    doc = torch.cat([torch.full((len(d),), k) for k, d in enumerate(row_ids)])
    allowed = torch.ones(L, L, dtype=torch.bool).tril()
    if doc_mask:
        allowed &= doc[:, None] == doc[None, :]
    mask4d = torch.zeros(1, 1, L, L).masked_fill(~allowed, torch.finfo(torch.float32).min)
    pos = torch.cat([torch.arange(len(d)) for d in row_ids]) if reset_pos else torch.arange(L)
    lg = model(input_ids=torch.tensor([ids]), attention_mask=mask4d, position_ids=pos[None]).logits[0].float()
    nll = F.cross_entropy(lg[:-1], torch.tensor(ids[1:]), reduction="none")
    out, start = [], 0
    for d in row_ids:
        out.append(nll[start:start + len(d) - 1])   # entry k predicts token k+1 of the row
        start += len(d)
    return out
