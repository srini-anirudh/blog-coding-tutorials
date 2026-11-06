"""A small character-level Transformer whose stability knobs are all switches.

    norm      "pre"  x + F(LN(x))           (GPT-2 onwards)
              "post" LN(x + F(x))           (the original Transformer)
              "deep" LN(alpha * x + F(x))   (DeepNorm, with beta-scaled init)
    init      "default" N(0, 0.02); "scaled" residual outputs at 0.02 / sqrt(2L); "xavier" Xavier-normal (gain 1)
              on every block matrix, as in Xiong et al.; "deepnorm" Xavier with beta
    qk_norm   LayerNorm on each head's queries and keys before the dot product
    softcap   z <- c * tanh(z / c) on the attention logits

Each block records the largest attention logit and the mean attention entropy of its last forward pass,
which are the two cheap logs Part VI of the post asks for.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LAB

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class Config:
    vocab: int = 65
    d: int = 128
    layers: int = 4
    heads: int = 4
    seq: int = 64
    norm: str = "pre"
    init: str = "default"
    qk_norm: bool = False
    softcap: float | None = None


def deepnorm_constants(layers: int) -> tuple[float, float]:
    """DeepNet's decoder-only constants: alpha = (2M)^(1/4), beta = (8M)^(-1/4)."""
    return (2 * layers) ** 0.25, (8 * layers) ** -0.25


class Block(nn.Module):
    def __init__(self, c: Config, alpha: float = 1.0) -> None:
        super().__init__()
        self.c, self.alpha, self.hd = c, alpha, c.d // c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.q, self.k, self.v, self.o = (nn.Linear(c.d, c.d, bias=False) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)
        if c.qk_norm:
            self.q_norm, self.k_norm = nn.LayerNorm(self.hd), nn.LayerNorm(self.hd)
        self.max_logit = 0.0
        self.entropy = 0.0

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        split = lambda y: y.view(b, t, self.c.heads, self.hd).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        if self.c.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        z = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        if self.c.softcap:
            z = self.c.softcap * torch.tanh(z / self.c.softcap)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        with torch.no_grad():
            self.max_logit = z.masked_fill(~mask, 0).abs().max().item()
        p = z.masked_fill(~mask, float("-inf")).softmax(-1)
        with torch.no_grad():
            self.entropy = -(p * p.clamp_min(1e-30).log()).sum(-1).mean().item()
        return self.o((p @ v).transpose(1, 2).reshape(b, t, -1))

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c.norm == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.ln1(self.alpha * x + self.attn(x))
        return self.ln2(self.alpha * x + self.mlp(x))


class TinyTransformer(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        if c.norm not in ("pre", "post", "deep"):
            raise ValueError(c.norm)
        self.c = c
        alpha, self.beta = deepnorm_constants(c.layers) if c.norm == "deep" else (1.0, 1.0)
        self.tok, self.pos = nn.Embedding(c.vocab, c.d), nn.Embedding(c.seq, c.d)
        self.blocks = nn.ModuleList(Block(c, alpha) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.d) if c.norm == "pre" else nn.Identity()
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        c = self.c
        scheme = "deepnorm" if c.norm == "deep" else c.init
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:
            if scheme == "xavier":
                for m in (blk.q, blk.k, blk.v, blk.o, blk.fc1, blk.fc2):
                    nn.init.xavier_normal_(m.weight)
            elif scheme == "scaled":
                for m in (blk.o, blk.fc2):
                    nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * c.layers))
            elif scheme == "deepnorm":
                # beta on the value, output and FFN weights; queries and keys keep gain 1 (DeepNet Lemma 4.1).
                for m in (blk.q, blk.k):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                for m in (blk.v, blk.o, blk.fc1, blk.fc2):
                    nn.init.xavier_normal_(m.weight, gain=self.beta)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(idx))

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def attention_logs(self) -> tuple[float, float]:
        """(largest |logit| over all layers, smallest mean entropy over all layers) from the last forward."""
        return max(b.max_logit for b in self.blocks), min(b.entropy for b in self.blocks)


# --- Data -----------------------------------------------------------------------------------------------------

def load_corpus() -> str:
    """Tiny Shakespeare. Looks in this lab's data/ and in c04-llm/data/, then downloads into this lab's data/."""
    for path in (LAB / "data" / "tinyshakespeare.txt", LAB.parents[1] / "c04-llm" / "data" / "tinyshakespeare.txt"):
        if path.exists():
            return path.read_text()
    path = LAB / "data" / "tinyshakespeare.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(CORPUS_URL, path)
    return path.read_text()


def synthetic_corpus(n: int = 20_000, seed: int = 0) -> str:
    """A small offline stand-in for tests: repeated words from a fixed vocabulary."""
    g = torch.Generator().manual_seed(seed)
    words = ["the ", "king ", "and ", "queen ", "of ", "rome ", "speak ", "now ", "\n"]
    picks = torch.randint(len(words), (n // 4,), generator=g)
    return "".join(words[i] for i in picks.tolist())[:n]


class CharData:
    def __init__(self, text: str, seq: int, val_frac: float = 0.1) -> None:
        chars = sorted(set(text))
        self.vocab = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        cut = int(len(ids) * (1 - val_frac))
        self.train, self.val, self.seq = ids[:cut], ids[cut:], seq

    def batch(self, size: int, gen: torch.Generator, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.train if split == "train" else self.val
        start = torch.randint(len(ids) - self.seq - 1, (size,), generator=gen)
        x = torch.stack([ids[s:s + self.seq] for s in start.tolist()])
        y = torch.stack([ids[s + 1:s + self.seq + 1] for s in start.tolist()])
        return x, y


# --- Training ---------------------------------------------------------------------------------------------------

def train(model: TinyTransformer, data: CharData, steps: int, lr: float, warmup: int = 0, batch: int = 32,
          clip: float | None = 1.0, seed: int = 0, log_every: int = 50, weight_decay: float = 0.0) -> dict:
    """AdamW with linear warmup then constant LR. Stops early if the loss goes non-finite."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    hist = {"step": [], "loss": [], "max_logit": [], "entropy": [], "grad_norm": [], "diverged": False}
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, step / warmup) if warmup else lr
        x, y = data.batch(batch, gen)
        loss = model.loss(x, y)
        if not torch.isfinite(loss):
            hist["diverged"] = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf")).item()
        opt.step()
        if step % log_every == 0 or step == 1:
            ml, ent = model.attention_logs()
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            hist["max_logit"].append(ml)
            hist["entropy"].append(ent)
            hist["grad_norm"].append(gnorm)
    return hist


@torch.no_grad()
def evaluate(model: TinyTransformer, data: CharData, batches: int = 20, batch: int = 32, seed: int = 123) -> float:
    gen = torch.Generator().manual_seed(seed)
    losses = [model.loss(*data.batch(batch, gen, "val")).item() for _ in range(batches)]
    return sum(losses) / len(losses)
