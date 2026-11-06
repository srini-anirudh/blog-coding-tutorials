"""A tiny vision-language model: image slots become prefix tokens for a causal transformer.

Shapes
    B  batch, K object slots (8), F features per slot (8), T response tokens, d model width, V vocab.

The image tokens come first, so every response token can attend to the whole image. During SFT a
fraction of examples see the null image, so log P(y | x, null) is a real language prior and the
split log P(y | x, I) = log P(y | x) + g(y; I | x) is computable for this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .world import BOS, DESCRIBE, EOS, F, K, PAD, PROMPT_LEN, VOCAB


class TinyVLM(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4, max_len: int = 48, reward_head: bool = False):
        super().__init__()
        self.img_proj = nn.Linear(F, d)
        self.slot = nn.Embedding(K, d)
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1 if reward_head else len(VOCAB))

    def hidden(self, img: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """img (B, K, F), ids (B, T) -> (B, T, d): hidden states at the text positions."""
        T = ids.shape[1]
        x = torch.cat([self.img_proj(img) + self.slot.weight,
                       self.tok(ids) + self.pos(torch.arange(T, device=ids.device))], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(K + T, device=ids.device)
        return self.norm(self.blocks(x, mask=mask, is_causal=True))[:, K:]

    def forward(self, img: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(img, ids))


def pad(seqs: list[list[int]]) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (T - len(s)) for s in seqs], dtype=torch.long)


def response_mask(ids: torch.Tensor) -> torch.Tensor:
    """(B, T-1) True where target ids[:, 1:] is a response token (not the prompt, not padding)."""
    tgt = ids[:, 1:]
    pos = torch.arange(1, ids.shape[1], device=ids.device)
    return (tgt != PAD) & (pos >= PROMPT_LEN)


def token_logprobs(model: TinyVLM, img: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(B, T-1) log p(ids[t] | image, ids[<t]), zero outside the response."""
    logp = F_.log_softmax(model(img, ids[:, :-1]).float(), -1)
    lp = logp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp * response_mask(ids)


def seq_logprob(model, img, ids, weights: torch.Tensor | None = None) -> torch.Tensor:
    """(B,) log pi(y | x, I). With `weights` (B, T-1), a weighted average as in RLHF-V's Dense DPO."""
    lp = token_logprobs(model, img, ids)
    if weights is None:
        return lp.sum(-1)
    w = weights * response_mask(ids)
    return (lp * w).sum(-1) / w.sum(-1).clamp_min(1e-8)


def sft_loss(model, img, ids) -> torch.Tensor:
    m = response_mask(ids)
    return -(token_logprobs(model, img, ids).sum() / m.sum())


@torch.no_grad()
def sample(model: TinyVLM, img: torch.Tensor, max_new: int = 30, temperature: float = 1.0,
           greedy: bool = False, gen: torch.Generator | None = None) -> list[list[int]]:
    """Decode B responses. No KV cache: the model is tiny and sequences are short."""
    B = img.shape[0]
    ids = torch.tensor([[BOS, DESCRIBE]] * B, device=img.device)
    done = torch.zeros(B, dtype=torch.bool, device=img.device)
    for _ in range(max_new):
        logits = model(img, ids)[:, -1].float()
        logits[:, PAD] = -1e9
        logits[:, BOS] = -1e9
        logits[:, DESCRIBE] = -1e9
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = F_.softmax(logits / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    out = []
    for row in ids.tolist():
        row = [t for t in row if t != PAD]
        out.append(row if row[-1] == EOS else row + [EOS])
    return out
