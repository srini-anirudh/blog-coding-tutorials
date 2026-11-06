"""Image tokenizers: an autoencoder with a quantizer in the middle.

The encoder downsamples a 16 x 16 image by f = 4 to a 4 x 4 grid, so each image becomes 16 tokens.
Three quantizers share that encoder and decoder:

- `VectorQuantizer`: the VQ-VAE codebook, with the straight-through copy and the commitment loss
      k* = argmin_k || z_e - e_k ||,   L_vq = || sg[z_e] - z_q ||^2 + beta || z_e - sg[z_q] ||^2
- `VectorQuantizer(code_dim=8, normalize=True)`: the LlamaGen fix, a low-dimensional, l2-normalised lookup
- `FSQ`: finite scalar quantization. Round each of a few dimensions to fixed levels. No codebook to collapse.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def token_budget(h: int, w: int, f: int, codebook: int, channels: int = 3, bits_per_value: int = 8) -> dict:
    """How many tokens an image costs, and the most information they can carry."""
    tokens = (h // f) * (w // f)
    bits_per_token = math.log2(codebook)
    raw_bits = h * w * channels * bits_per_value
    return {
        "tokens": tokens,
        "bits_per_token": bits_per_token,
        "max_bits": tokens * bits_per_token,
        "raw_bits": raw_bits,
        "compression": raw_bits / (tokens * bits_per_token),
    }


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Forward pass uses q. Backward pass copies the gradient straight to x."""
    return x + (q - x).detach()


class VectorQuantizer(nn.Module):
    def __init__(self, codebook: int, dim: int, beta: float = 0.25, code_dim: int | None = None, normalize: bool = False):
        super().__init__()
        self.codebook, self.beta, self.normalize = codebook, beta, normalize
        self.code_dim = code_dim or dim
        self.down = nn.Linear(dim, self.code_dim) if code_dim else nn.Identity()
        self.up = nn.Linear(self.code_dim, dim) if code_dim else nn.Identity()
        self.embed = nn.Embedding(codebook, self.code_dim)
        nn.init.uniform_(self.embed.weight, -1 / codebook, 1 / codebook)

    def forward(self, z: torch.Tensor):
        """z: (n, tokens, dim). Returns quantized z, indices (n, tokens) and the codebook + commitment loss."""
        ze = self.down(z)
        codes = self.embed.weight
        if self.normalize:
            ze, codes = F.normalize(ze, dim=-1), F.normalize(codes, dim=-1)
        dist = (ze.pow(2).sum(-1, keepdim=True) - 2 * ze @ codes.t() + codes.pow(2).sum(-1))
        idx = dist.argmin(-1)
        zq = codes[idx]
        loss = F.mse_loss(zq, ze.detach()) + self.beta * F.mse_loss(ze, zq.detach())
        return self.up(ste(ze, zq)), idx, loss


class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al., 2023). The implicit codebook size is prod(levels)."""

    def __init__(self, levels: list[int], dim: int):
        super().__init__()
        self.levels = torch.tensor(levels)
        self.codebook = int(torch.prod(self.levels))
        self.down = nn.Linear(dim, len(levels))
        self.up = nn.Linear(len(levels), dim)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), 0)
        self.register_buffer("basis", basis)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self.levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z: torch.Tensor):
        zb = self.bound(self.down(z))
        zq = ste(zb, zb.round())
        idx = ((zq.round() + self.levels // 2).long() * self.basis).sum(-1)
        return self.up(zq / (self.levels // 2)), idx, torch.zeros(())


class Tokenizer(nn.Module):
    """Conv encoder (16 x 16 -> 4 x 4 x dim), a quantizer, conv decoder back to pixels."""

    def __init__(self, quantizer: nn.Module, dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, dim, 4, stride=2, padding=1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.quantizer = quantizer
        self.dim = dim

    def encode(self, x: torch.Tensor):
        z = self.enc(x)  # (n, dim, 4, 4)
        n, d, h, w = z.shape
        zq, idx, qloss = self.quantizer(z.flatten(2).transpose(1, 2))
        return zq, idx, qloss, (h, w)

    def decode_latent(self, zq: torch.Tensor, hw: tuple[int, int] = (4, 4)) -> torch.Tensor:
        n = zq.shape[0]
        return self.dec(zq.transpose(1, 2).reshape(n, self.dim, *hw))

    def forward(self, x: torch.Tensor):
        zq, idx, qloss, hw = self.encode(x)
        recon = self.decode_latent(zq, hw)
        return recon, idx, F.mse_loss(recon, x) + qloss

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Turn token ids (n, 16) back into images. Works for both quantizer types."""
        q = self.quantizer
        if isinstance(q, FSQ):
            digits = (idx[..., None] // q.basis) % q.levels
            zq = q.up((digits - q.levels // 2).float() / (q.levels // 2))
        else:
            codes = q.embed.weight
            if q.normalize:
                codes = F.normalize(codes, dim=-1)
            zq = q.up(codes[idx])
        side = int(math.isqrt(idx.shape[1]))
        return self.decode_latent(zq, (side, side))


def codebook_stats(idx: torch.Tensor, codebook: int) -> dict:
    """Usage = share of entries chosen at least once. Perplexity = effective number of entries in use."""
    counts = torch.bincount(idx.flatten(), minlength=codebook).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    return {"usage": float((counts > 0).float().mean()), "perplexity": float(torch.exp(-(nz * nz.log()).sum()))}


def train_tokenizer(tok: Tokenizer, images: torch.Tensor, steps: int, batch: int = 128, lr: float = 2e-3, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(tok.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x = images[torch.randint(0, len(images), (batch,))]
        _, _, loss = tok(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def tokenize(tok: Tokenizer, images: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    return torch.cat([tok.encode(images[i : i + batch])[1] for i in range(0, len(images), batch)])
