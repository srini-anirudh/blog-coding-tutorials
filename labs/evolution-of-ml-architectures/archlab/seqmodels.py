"""Five ways to mix positions, behind one interface: RNN, LSTM, causal attention, LTI SSM, selective SSM.

Every model is Embedding -> n_layers x [x + Mixer(LayerNorm(x))] -> LayerNorm -> Linear, read at the last
position. Only the mixer changes, which is the post's claim: the mixing operator is the architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .scan import parallel_scan, sequential_scan


class RecurrentMixer(nn.Module):
    def __init__(self, d: int, kind: str):
        super().__init__()
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind](d, d, batch_first=True)
        if kind == "lstm":  # forget-gate bias +1: start with an open gradient path (post, "Why gates fix it")
            for name, p in self.rnn.named_parameters():
                if name.startswith("bias_ih"):
                    p.data[d:2 * d] = 1.0

    def forward(self, x):
        return self.rnn(x)[0]


class AttentionMixer(nn.Module):
    """Causal multi-head self-attention. Position comes from the learned embedding added in SeqModel."""

    def __init__(self, d: int, n_heads: int = 2):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class SSMMixer(nn.Module):
    """A Mamba-1 block. selective=True makes Delta, B, C functions of x_t; False makes them fixed (S4D-like LTI).

    in_proj -> (x, z); x -> causal depthwise conv -> SiLU -> SSM; y * SiLU(z) -> out_proj.
    scan="parallel" uses `parallel_scan` (ceil(log2 N) rounds, about log2 N times the work). scan="sequential"
    runs N small steps. Both give the same numbers; on a CPU the sequential one is faster, because a CPU has
    no thousands of idle lanes for the extra work to fill. Training here uses "sequential".
    """

    def __init__(self, d: int, E: int = 2, s: int = 8, conv: int = 4, selective: bool = True, scan: str = "sequential"):
        super().__init__()
        self.scan = scan
        e, r = E * d, math.ceil(d / 16)
        self.e, self.s, self.r, self.selective = e, s, r, selective
        self.in_proj = nn.Linear(d, 2 * e, bias=False)
        self.conv = nn.Conv1d(e, e, conv, groups=e, padding=conv - 1)
        if selective:
            self.x_proj = nn.Linear(e, r + 2 * s, bias=False)
            self.dt_proj = nn.Linear(r, e)
            dt_bias = self.dt_proj.bias
        else:
            self.B = nn.Parameter(torch.randn(e, s) / math.sqrt(s))
            self.C = nn.Parameter(torch.randn(e, s) / math.sqrt(s))
            dt_bias = self.dt_bias = nn.Parameter(torch.empty(e))
        # Delta starts log-uniform in [1e-3, 1e-1]; the bias stores softplus^{-1}(Delta).
        dt = torch.exp(torch.rand(e) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, s + 1, dtype=torch.float32)).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d, bias=False)
        self.last_dt = None  # (B, N, e), kept for the gate analysis

    def forward(self, u):
        b, n, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :n].transpose(1, 2))  # (B, N, e)
        A = -torch.exp(self.A_log)  # (e, s), negative: every a_bar is in (0, 1)
        if self.selective:
            dt, Bm, Cm = self.x_proj(x).split([self.r, self.s, self.s], dim=-1)
            dt = F.softplus(self.dt_proj(dt))  # (B, N, e)
            Bm, Cm = Bm[:, :, None, :], Cm[:, :, None, :]  # (B, N, 1, s)
        else:
            dt = F.softplus(self.dt_bias).expand(b, n, self.e)
            Bm, Cm = self.B, self.C  # (e, s)
        self.last_dt = dt.detach()
        a = torch.exp(dt[..., None] * A)  # (B, N, e, s): the input-dependent forget gate
        b_bar = dt[..., None] * Bm * x[..., None]
        h = parallel_scan(a, b_bar)[0] if self.scan == "parallel" else sequential_scan(a, b_bar)
        y = (h * Cm).sum(-1) + self.D * x
        return self.out_proj(y * F.silu(z))


MIXERS = {
    "rnn": lambda d: RecurrentMixer(d, "rnn"),
    "lstm": lambda d: RecurrentMixer(d, "lstm"),
    "attention": lambda d: AttentionMixer(d),
    "lti_ssm": lambda d: SSMMixer(d, selective=False),
    "selective_ssm": lambda d: SSMMixer(d, selective=True),
}


class SeqModel(nn.Module):
    def __init__(self, kind: str, vocab: int, n_classes: int, d: int = 64, n_layers: int = 2, max_len: int = 1024):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d) if kind == "attention" else None
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.mixers = nn.ModuleList(MIXERS[kind](d) for _ in range(n_layers))
        self.final = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, tokens):
        x = self.emb(tokens)
        if self.pos is not None:
            x = x + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for norm, mix in zip(self.norms, self.mixers):
            x = x + mix(norm(x))
        return self.head(self.final(x[:, -1]))


def key_recall_batch(batch: int, n_filler: int, n_keys: int = 16, generator: torch.Generator | None = None):
    """[MARK, key, n_filler tokens drawn from the same key vocabulary, QUERY] -> key.

    The filler looks exactly like keys, so a model must remember *which* token came after MARK, not just any
    key-like token. MARK = n_keys, QUERY = n_keys + 1. Chance accuracy is 1 / n_keys.
    """
    key = torch.randint(0, n_keys, (batch,), generator=generator)
    filler = torch.randint(0, n_keys, (batch, n_filler), generator=generator)
    mark = torch.full((batch, 1), n_keys)
    query = torch.full((batch, 1), n_keys + 1)
    return torch.cat([mark, key[:, None], filler, query], dim=1), key
