"""Priors over image tokens, and two orders for sampling them.

The sequence is one condition token (the shape class, or a "null" class) followed by the 16 image tokens.

- Raster autoregression: a causal transformer predicts token i from tokens < i. One forward pass per token.
- MaskGIT: a bidirectional transformer predicts every masked token at once. Keep the most confident
  predictions, re-mask the rest, repeat. A handful of passes for the whole grid.

Both are trained with condition dropout, so they can also be sampled with classifier-free guidance:
    l_g = l_u + s (l_c - l_u)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Trunk, causal_mask


def cfg_logits(l_uncond: torch.Tensor, l_cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-free guidance on logits. scale = 1 gives the conditional logits back."""
    return l_uncond + scale * (l_cond - l_uncond)


def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1) / math.log(2)


class TokenPrior(nn.Module):
    def __init__(self, vocab: int, n_tokens: int, n_classes: int, order: str, d: int = 128, layers: int = 3):
        super().__init__()
        assert order in ("raster", "maskgit")
        self.vocab, self.n_tokens, self.n_classes, self.order = vocab, n_tokens, n_classes, order
        self.mask_id = vocab
        self.null_class = n_classes
        self.tok = nn.Embedding(vocab + 1, d)
        self.cls = nn.Embedding(n_classes + 1, d)
        self.trunk = Trunk(n_tokens + 1, d, layers)
        self.head = nn.Linear(d, vocab)
        self.passes = 0  # forward passes of the network, counted by sample_*

    def logits(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Logits for the 16 image positions, shape (n, n_tokens, vocab)."""
        self.passes += 1
        if self.order == "raster":
            h = torch.cat([self.cls(cond)[:, None], self.tok(tokens[:, :-1])], dim=1)
            out = self.trunk(h, causal_mask(h.shape[1]))
            return self.head(out)
        h = torch.cat([self.cls(cond)[:, None], self.tok(tokens)], dim=1)
        return self.head(self.trunk(h)[:, 1:])

    def loss(self, tokens: torch.Tensor, cond: torch.Tensor, p_uncond: float = 0.1) -> torch.Tensor:
        drop = torch.rand(cond.shape[0]) < p_uncond
        cond = torch.where(drop, torch.full_like(cond, self.null_class), cond)
        if self.order == "raster":
            return F.cross_entropy(self.logits(tokens, cond).flatten(0, 1), tokens.flatten())
        # MaskGIT training: mask a cosine-scheduled share of tokens, predict only those.
        n, L = tokens.shape
        ratio = torch.cos(torch.rand(n) * math.pi / 2)
        masked = torch.rand(n, L) < ratio[:, None].clamp(min=1.0 / L)
        inp = torch.where(masked, torch.full_like(tokens, self.mask_id), tokens)
        logits = self.logits(inp, cond)
        return F.cross_entropy(logits[masked], tokens[masked])

    def _guided(self, tokens: torch.Tensor, cond: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return self.logits(tokens, cond)
        both = self.logits(torch.cat([tokens, tokens]), torch.cat([cond, torch.full_like(cond, self.null_class)]))
        l_cond, l_uncond = both.chunk(2)
        return cfg_logits(l_uncond, l_cond, scale)

    @torch.no_grad()
    def sample_raster(self, cond: torch.Tensor, scale: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        n = cond.shape[0]
        tokens = torch.zeros(n, self.n_tokens, dtype=torch.long)
        for i in range(self.n_tokens):  # no KV cache: each step re-reads the prefix, one pass per token
            logits = self._guided(tokens, cond, scale)[:, i] / temperature
            tokens[:, i] = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        return tokens

    @torch.no_grad()
    def sample_maskgit(self, cond: torch.Tensor, steps: int = 4, scale: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        n, L = cond.shape[0], self.n_tokens
        tokens = torch.full((n, L), self.mask_id, dtype=torch.long)
        for t in range(steps):
            masked = tokens == self.mask_id
            logits = self._guided(tokens, cond, scale) / temperature
            probs = F.softmax(logits, -1)
            draw = torch.multinomial(probs.flatten(0, 1), 1).view(n, L)
            conf = torch.gather(probs, -1, draw[..., None]).squeeze(-1).log()
            conf = conf + (1 - (t + 1) / steps) * -torch.log(-torch.log(torch.rand(n, L)))  # annealed Gumbel noise
            conf = torch.where(masked, conf, torch.full_like(conf, float("inf")))  # never re-mask a kept token
            n_keep_masked = int(math.floor(L * math.cos(math.pi / 2 * (t + 1) / steps)))
            new = torch.where(masked, draw, tokens)
            if n_keep_masked > 0:
                remask = conf.argsort(dim=1)[:, :n_keep_masked]
                new.scatter_(1, remask, self.mask_id)
            tokens = new
        return tokens


def train_prior(prior: TokenPrior, tokens: torch.Tensor, cond: torch.Tensor, steps: int, batch: int = 256, lr: float = 1e-3, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(steps):
        i = torch.randint(0, len(tokens), (batch,))
        loss = prior.loss(tokens[i], cond[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def pass_counts(grid: int, maskgit_steps: int, scales: list[int], diffusion_steps: int) -> dict:
    """Sequential forward passes to fill a grid x grid token image, per decoding order (post Figure 2)."""
    return {
        "raster autoregression": grid * grid,
        f"masked parallel decoding ({maskgit_steps} rounds)": maskgit_steps,
        f"next-scale prediction (scales {scales})": len(scales),
        f"diffusion ({diffusion_steps} steps)": diffusion_steps,
    }
