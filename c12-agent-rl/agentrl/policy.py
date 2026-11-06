"""The policy: a tiny decoder-only transformer with a value head.

It is a language model in every way that matters for agent RL. It reads one token sequence (task, actions,
observations), predicts the next token, and generates an action token by token until ``<eos>``. So the rollout,
the BC loss and the GRPO loss are the same code that would train SmolLM2 or a 7B model; only the size changes.

The block is a trimmed version of the C4 design (pre-norm, RMSNorm, causal SDPA attention, GELU MLP). Learned
absolute positions are enough at 256 tokens. The value head is one linear layer on the final hidden state; only the
action-level (critic) variant of the update uses it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_len: int = 256


@dataclass
class PolicyOutput:
    logits: torch.Tensor   # (B, T, V)
    values: torch.Tensor   # (B, T)  estimate of the final reward from this prefix


class Block(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).unbind(2)
        att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             is_causal=True)          # (B, H, T, d_h)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.norm2(x))


class TinyPolicy(nn.Module):
    """Right-padded batches need no attention mask: with causal attention, position t never sees the padding
    that comes after the sequence ends, so the logits at real positions are exact."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> PolicyOutput:
        """input_ids: (B, T) -> logits (B, T, V), values (B, T)."""
        T = input_ids.shape[1]
        x = self.tok(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        for block in self.blocks:
            x = block(x)
        h = self.norm(x)
        return PolicyOutput(self.lm_head(h), self.value_head(h).squeeze(-1))

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TinyPolicy":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(PolicyConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """log pi(tokens[:, t] | tokens[:, :t]) for t >= 1, aligned so out[:, t] scores tokens[:, t]; out[:, 0] = 0.

    logits: (B, T, V) from a forward over ``tokens`` (B, T). Returns (B, T).
    Dividing by the sampling temperature keeps the training distribution equal to the one we sampled from.
    """
    logp = F.log_softmax(logits[:, :-1].float() / temperature, dim=-1)
    picked = logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, T-1)
    return F.pad(picked, (1, 0))
