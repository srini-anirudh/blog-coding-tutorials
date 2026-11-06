"""Three ways to hand over a location, each as a small PyTorch model for the toy referring task.

Type    TypedGrounder writes the box as tokens: four location tokens, or twelve digit tokens (three per coordinate).
Select  SelectGrounder scores region tokens pooled from a class-agnostic proposer and picks one (Groma style).
Decode  DecodeGrounder emits a <SEG> token whose hidden state prompts a mask decoder (LISA style).

The image encoders are a few convolutions, the "language model" is a two-layer Transformer, and the mask decoder
is a dot product between the prompt and per-pixel features. SAM is far larger, but the interface is the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .toy import COLOUR_NAMES, SIZE, bins_to_box, box_to_bins


def _coords(x: torch.Tensor) -> torch.Tensor:
    """Append x and y coordinate channels (CoordConv), so convolutions know where they are."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, xs, ys], 1)


def _conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())


class PixelEncoder(nn.Module):
    """Image -> coarse tokens (B, C, 16, 16) and, if asked, pixel features (B, D, 32, 32).

    The stem is a stride-2 patchify convolution, so the full-resolution work stays small on a CPU.
    """

    def __init__(self, in_ch: int = 3, c: int = 64, d: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch + 2, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU())
        self.down = nn.Sequential(_conv(32, c, 2), _conv(c, c))
        self.up = nn.Sequential(nn.Conv2d(c + 32, 48, 1), nn.GELU(), nn.Conv2d(48, d, 1))

    def forward(self, x, pixels: bool = True):
        s = self.stem(_coords(x))
        t = self.down(s)
        if not pixels:
            return t, None
        u = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return t, self.up(torch.cat([u, s], 1))


def upsample(logits: torch.Tensor) -> torch.Tensor:
    """(B, 32, 32) mask logits or features -> full 64 x 64 resolution."""
    x = logits if logits.dim() == 4 else logits[:, None]
    y = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return y if logits.dim() == 4 else y[:, 0]


# ---- Type ------------------------------------------------------------------------------------

class SeqDecoder(nn.Module):
    """A GRU that writes `steps` tokens from a vocabulary of `vocab`, starting from an image-and-query state."""

    def __init__(self, vocab: int, steps: int, hidden: int = 256):
        super().__init__()
        self.vocab, self.steps = vocab, steps
        self.emb = nn.Embedding(vocab + 1, 64)        # last id is <bos>
        self.pos = nn.Embedding(steps, 64)
        self.gru = nn.GRUCell(64, hidden)
        self.out = nn.Linear(hidden, vocab)

    def _step(self, h, tok, t):
        x = self.emb(tok) + self.pos.weight[t]
        h = self.gru(x, h)
        return h, self.out(h)

    def nll(self, h, targets):
        """Teacher-forced negative log-likelihood of each sequence, summed over its tokens. (B,)"""
        tok = torch.full((h.shape[0],), self.vocab, dtype=torch.long, device=h.device)
        total = 0
        for t in range(self.steps):
            h, logits = self._step(h, tok, t)
            total = total + F.cross_entropy(logits, targets[:, t], reduction="none")
            tok = targets[:, t]
        return total

    def generate(self, h, sample: bool = False):
        """Greedy (or sampled) decoding. Returns tokens (B, steps) and their summed log-probability (B,)."""
        tok = torch.full((h.shape[0],), self.vocab, dtype=torch.long, device=h.device)
        toks, logp = [], 0
        for t in range(self.steps):
            h, logits = self._step(h, tok, t)
            lp = logits.log_softmax(-1)
            tok = torch.multinomial(lp.exp(), 1).squeeze(1) if sample else lp.argmax(-1)
            logp = logp + lp.gather(1, tok[:, None]).squeeze(1)
            toks.append(tok)
        return torch.stack(toks, 1), logp


class TypedGrounder(nn.Module):
    """Writes the box on a 0-999 grid, as 4 location tokens (fmt='loc') or as 12 digits (fmt='digits')."""

    def __init__(self, fmt: str = "loc", bins: int = 1000, hidden: int = 256):
        super().__init__()
        assert fmt in ("loc", "digits")
        self.fmt, self.bins = fmt, bins
        self.enc = PixelEncoder(3 + len(COLOUR_NAMES))
        self.pool = nn.Sequential(nn.Conv2d(64, 32, 3, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(32 * 8 * 8, hidden), nn.GELU())
        self.dec = SeqDecoder(bins if fmt == "loc" else 10, 4 if fmt == "loc" else 12, hidden)

    def state(self, image, query):
        q = F.one_hot(query, len(COLOUR_NAMES)).float()[:, :, None, None].expand(-1, -1, SIZE, SIZE)
        t, _ = self.enc(torch.cat([image, q], 1), pixels=False)
        return self.pool(t)

    def to_tokens(self, box):
        k = box_to_bins(box, self.bins)
        if self.fmt == "loc":
            return k
        return torch.stack([k // 100, (k // 10) % 10, k % 10], -1).flatten(1)

    def to_bins(self, toks):
        if self.fmt == "loc":
            return toks
        d = toks.view(-1, 4, 3)
        return d[..., 0] * 100 + d[..., 1] * 10 + d[..., 2]

    def nll(self, image, query, box):
        return self.dec.nll(self.state(image, query), self.to_tokens(box))

    def predict(self, image, query, sample: bool = False):
        toks, logp = self.dec.generate(self.state(image, query), sample)
        return bins_to_box(self.to_bins(toks), self.bins), logp


# ---- Select ----------------------------------------------------------------------------------

class SelectGrounder(nn.Module):
    """Pools one region token per proposal, then scores each against the query. Never writes a coordinate."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.enc = PixelEncoder(3, d=d)
        self.region = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64))
        self.query = nn.Embedding(len(COLOUR_NAMES), 64)

    def scores(self, image, query, props):
        """props: one list of (H, W) bool masks per image. Returns (B, R) scores, -inf where padded."""
        _, pix = self.enc(image)
        pix = upsample(pix)
        r = max(1, max(len(p) for p in props))
        m = torch.zeros(len(props), r, SIZE, SIZE)
        for i, regions in enumerate(props):
            if regions:
                m[i, :len(regions)] = torch.stack(regions).float()
        m = m.to(pix.device)
        area = m.sum((2, 3))
        tok = torch.einsum("brhw,bdhw->brd", m, pix) / area.clamp(min=1)[..., None]  # mask-average pooling
        s = (self.region(tok) * self.query(query)[:, None]).sum(-1)
        return s.masked_fill(area == 0, float("-inf"))


# ---- Decode ----------------------------------------------------------------------------------

class DecodeGrounder(nn.Module):
    """LISA in miniature: p = MLP(h_SEG), mask = Dec(F_img, p). The mask loss trains the whole chain."""

    def __init__(self, d: int = 32, width: int = 64, layers: int = 2):
        super().__init__()
        self.enc = PixelEncoder(3, c=width, d=d)
        self.pos = nn.Parameter(torch.randn(1, 16 * 16, width) * 0.02)
        self.query = nn.Embedding(len(COLOUR_NAMES), width)
        self.seg = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.lm = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.to_prompt = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, d))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, image, query):
        t, pix = self.enc(image)
        img_tok = t.flatten(2).transpose(1, 2) + self.pos                  # (B, 256, width)
        seq = torch.cat([img_tok, self.query(query)[:, None], self.seg.expand(len(query), -1, -1)], 1)
        h_seg = self.lm(seq)[:, -1]                                          # the <SEG> token's last hidden state
        p = self.to_prompt(h_seg)                                            # prompt for the mask decoder
        return upsample(torch.einsum("bd,bdhw->bhw", p, pix) / pix.shape[1] ** 0.5 + self.bias)


def bce_dice(logits, target):
    """L = L_bce + L_dice, the mask loss of LISA (both weights 1 here)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum((1, 2)) + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
    return bce + dice.mean()
