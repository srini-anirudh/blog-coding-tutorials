"""Megatron tensor parallelism, with optional sequence parallelism, for the reference Block.

Column-parallel: W = [W_1 | W_2], each rank computes its slice of the output features. No communication.
Row-parallel:    each rank holds the input-feature slice, computes a full-shaped partial sum; one reduction.
Attention: QKV is column-parallel by head, the attention itself is local, the output projection is row-parallel.

Without SP, activations between the parallel regions (layernorm input, residual) are replicated on every rank,
and the boundaries are f (enter) and f_bar (leave): one AllReduce forward per sub-block.
With SP, those regions are split along the sequence instead, and the boundaries become g (AllGather) and
g_bar (ReduceScatter): the same AllReduce, cut in half with the replicated region removed from between.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import comm
from .model import Block, attention


def _shard(t: torch.Tensor, dim: int | None, rank: int, n: int) -> nn.Parameter:
    p = nn.Parameter(t.detach().clone() if dim is None else t.detach().chunk(n, dim=dim)[rank].clone())
    p._tp_dim = dim  # how to reassemble the full tensor; None means replicated on every rank
    return p


class TPBlock(nn.Module):
    def __init__(self, ref: Block, group=None, sequence_parallel: bool = False, regather: bool = False):
        super().__init__()
        self.group, self.sp, self.regather = group, sequence_parallel, regather and sequence_parallel
        rank, n = dist.get_rank(group), dist.get_world_size(group)
        assert ref.heads % n == 0, "heads must divide by the TP degree"
        self.local_heads = ref.heads // n
        self.ln1, self.ln2 = nn.LayerNorm(ref.ln1.normalized_shape), nn.LayerNorm(ref.ln2.normalized_shape)
        for mine, theirs in ((self.ln1, ref.ln1), (self.ln2, ref.ln2)):
            mine.weight, mine.bias = _shard(theirs.weight, None, rank, n), _shard(theirs.bias, None, rank, n)
        # column-parallel: split output features (rows of the [out, in] weight)
        self.qkv_w, self.qkv_b = _shard(ref.qkv.weight, 0, rank, n), _shard(ref.qkv.bias, 0, rank, n)
        self.fc1_w, self.fc1_b = _shard(ref.fc1.weight, 0, rank, n), _shard(ref.fc1.bias, 0, rank, n)
        # row-parallel: split input features (columns); the bias is added once, after the reduction
        self.proj_w, self.proj_b = _shard(ref.proj.weight, 1, rank, n), _shard(ref.proj.bias, None, rank, n)
        self.fc2_w, self.fc2_b = _shard(ref.fc2.weight, 1, rank, n), _shard(ref.fc2.bias, None, rank, n)

    def _column(self, x, w, b):
        """Enter the TP region and run a column-parallel matmul. Plain SP saves the gathered full-sequence
        input for the weight gradient; with regather it saves only the shard and gathers it again in backward."""
        if self.regather:
            return _GatherLinear.apply(x, w, b, self.group)
        return F.linear(comm.g(x, 0, self.group) if self.sp else comm.f(x, self.group), w, b)

    def _leave(self, partial):  # row-parallel partial sums -> complete, replicated or sequence-sharded
        return comm.g_bar(partial, 0, self.group) if self.sp else comm.f_bar(partial, self.group)

    def forward(self, x):
        """x: [seq, batch, hidden], or [seq / T, batch, hidden] with sequence parallelism."""
        qkv = self._column(self.ln1(x), self.qkv_w, self.qkv_b)
        s, b, _ = qkv.shape
        q, k, v = qkv.view(s, b, self.local_heads, 3, -1).permute(3, 1, 2, 0, 4)
        o = attention(q, k, v).permute(2, 0, 1, 3).reshape(s, b, -1)
        x = x + self._leave(F.linear(o, self.proj_w)) + self.proj_b
        h = F.gelu(self._column(self.ln2(x), self.fc1_w, self.fc1_b))
        x = x + self._leave(F.linear(h, self.fc2_w)) + self.fc2_b
        return x

    def sync_replicated_grads(self):
        """Under SP each rank saw only its own tokens in the layernorms and the row-parallel biases, so their
        gradients are partial sums. Without SP they are already identical and nothing needs to move."""
        if not self.sp:
            return
        with comm.phase("grad_sync"):
            for p in self.parameters():
                if p._tp_dim is None and p.grad is not None:
                    comm.all_reduce(p.grad, self.group)

    def full_grads(self) -> dict[str, torch.Tensor]:
        """Reassemble every gradient to its single-device shape, for comparison with the reference."""
        n, out = dist.get_world_size(self.group), {}
        for name, p in self.named_parameters():
            if p._tp_dim is None:
                out[name] = p.grad.clone()
            else:
                parts = [torch.empty_like(p.grad) for _ in range(n)]
                dist.all_gather(parts, p.grad.contiguous(), group=self.group)
                out[name] = torch.cat(parts, dim=p._tp_dim)
        return out


class _GatherLinear(torch.autograd.Function):
    """AllGather(sequence) then a column-parallel matmul, keeping only the sequence shard for backward
    (Korthikanti et al. 2022). The price is a second AllGather in the backward pass."""

    @staticmethod
    def forward(ctx, x_shard, w, b, group):
        ctx.save_for_backward(x_shard, w)
        ctx.group = group
        return F.linear(comm.all_gather(x_shard, 0, group), w, b)

    @staticmethod
    def backward(ctx, dy):
        x_shard, w = ctx.saved_tensors
        with comm.phase("bwd"):
            full = comm.all_gather(x_shard, 0, ctx.group)  # the extra collective
            dx = comm.reduce_scatter(dy @ w, 0, ctx.group)
        dw = dy.reshape(-1, dy.shape[-1]).t() @ full.reshape(-1, full.shape[-1])
        return dx, dw, dy.reshape(-1, dy.shape[-1]).sum(0), None


REF_NAMES = {"ln1.weight": "ln1.weight", "ln1.bias": "ln1.bias", "ln2.weight": "ln2.weight", "ln2.bias": "ln2.bias",
             "qkv_w": "qkv.weight", "qkv_b": "qkv.bias", "proj_w": "proj.weight", "proj_b": "proj.bias",
             "fc1_w": "fc1.weight", "fc1_b": "fc1.bias", "fc2_w": "fc2.weight", "fc2_b": "fc2.bias"}


def mlp_row_first(x, fc1_w, fc1_b, fc2_w, group=None):
    """The wrong order: row-parallel first. The partial sums must be completed before GELU, because
    GELU(a + b) != GELU(a) + GELU(b), and the column-parallel second layer leaves its output split by
    feature, which needs an AllGather. Two collectives forward where column-then-row needs one."""
    rank, n = dist.get_rank(group), dist.get_world_size(group)
    x_t = x.chunk(n, dim=-1)[rank]                          # input features, sliced for free (x is replicated)
    y = comm.all_reduce(F.linear(x_t, fc1_w.chunk(n, dim=1)[rank]), group) + fc1_b  # complete before GELU
    z_t = F.linear(F.gelu(y), fc2_w.chunk(n, dim=0)[rank])  # output features split
    return comm.all_gather(z_t, -1, group)
