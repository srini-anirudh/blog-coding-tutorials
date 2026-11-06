"""Paged KV block manager: free list, block tables, reference counts, copy-on-write and prefix caching.

This file is the allocator from the post's "Paged KV cache", "Prefix caching" and "vLLM up close" sections.
It never touches a tensor. It hands out physical block numbers, and the model writes keys and values
into those blocks. A copy callback is the only link back to GPU memory (for copy-on-write).

Invariants (tests/test_block_manager.py checks them):
- every block is either in the free list (ref count 0) or referenced by at least one block table;
- a block's hash is registered in hash_to_block at most once;
- a freed block keeps its hash until the moment it is handed to someone else (lazy eviction).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Hashable, Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def block_hashes(
    tokens: Sequence[int],
    block_size: int,
    extra_keys: Sequence[Hashable] | None = None,
    chained: bool = True,
) -> list[int]:
    """One hash per full block of `tokens`.

    chained=True: hash(parent hash, this block's tokens, extra key). A key then means "this content,
    at this position, after this history". chained=False hashes the block alone, which is the bug the
    post warns about: two different conversations that share one 16-token block collide.
    extra_keys[i] is mixed into block i. A VLM puts the image content hash there, because image
    placeholder token ids are identical for every image.
    """
    out: list[int] = []
    parent = None
    for i in range(len(tokens) // block_size):
        chunk = tuple(tokens[i * block_size : (i + 1) * block_size])
        extra = extra_keys[i] if extra_keys is not None else None
        h = hash((parent, chunk, extra)) if chained else hash((chunk, extra))
        out.append(h)
        parent = h
    return out


class BlockManager:
    """Hands out fixed-size KV blocks to sequences, lazily, one block at a time."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        enable_prefix_caching: bool = False,
        chained_hash: bool = True,
        watermark: float = 0.0,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.chained_hash = chained_hash
        # Keep this many blocks free at admission, so running sequences can still grow.
        self.watermark_blocks = int(watermark * num_blocks)
        # Free list in eviction order: popitem(last=False) takes the least recently freed block.
        self.free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        self.ref = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.tables: dict[int, list[int]] = {}
        self.seq_hashes: dict[int, list[int]] = {}
        self.copy_fn: Callable[[int, int], None] | None = None  # (src, dst) copies KV on copy-on-write

    # ---- basic accounting -------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free)

    def usage(self) -> float:
        return 1.0 - self.num_free / self.num_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return cdiv(num_tokens, self.block_size)

    def _take_free(self) -> int:
        bid, _ = self.free.popitem(last=False)
        old = self.block_hash[bid]
        if old is not None:  # this is the real eviction: the cached prefix is gone only now
            del self.hash_to_block[old]
            self.block_hash[bid] = None
        self.ref[bid] = 1
        return bid

    def _release(self, bid: int) -> None:
        self.ref[bid] -= 1
        assert self.ref[bid] >= 0, f"block {bid} freed twice"
        if self.ref[bid] == 0:
            self.free[bid] = None  # tail of the free list: evicted last

    # ---- sequence lifetime ------------------------------------------------------------------
    def allocate_prompt(self, seq_id: int, tokens: Sequence[int], extra_keys=None) -> int:
        """Open a block table for a new sequence. Returns how many prompt tokens are already cached.

        No block is allocated for uncached tokens here: that happens in ensure() as tokens are computed.
        At least one token is always left to compute, because the model must produce its logits.
        """
        assert seq_id not in self.tables, f"sequence {seq_id} already has a block table"
        table: list[int] = []
        hashes: list[int] = []
        if self.enable_prefix_caching:
            hashes = block_hashes(tokens, self.block_size, extra_keys, self.chained_hash)
            for h in hashes[: (len(tokens) - 1) // self.block_size]:
                bid = self.hash_to_block.get(h)
                if bid is None:
                    break
                if self.ref[bid] == 0:
                    del self.free[bid]  # brought back from the dead
                self.ref[bid] += 1
                table.append(bid)
        self.tables[seq_id] = table
        self.seq_hashes[seq_id] = hashes[: len(table)]
        return len(table) * self.block_size

    def can_grow(self, seq_id: int, num_tokens: int, reserve: int = 0) -> bool:
        need = self.blocks_for(num_tokens) - len(self.tables[seq_id])
        return need <= self.num_free - reserve

    def ensure(self, seq_id: int, num_tokens: int, reserve: int = 0) -> bool:
        """Make the table cover num_tokens positions. All or nothing: False means out of blocks."""
        if not self.can_grow(seq_id, num_tokens, reserve):
            return False
        table = self.tables[seq_id]
        while len(table) < self.blocks_for(num_tokens):
            table.append(self._take_free())
        return True

    def prepare_write(self, seq_id: int, start: int, end: int) -> None:
        """Copy-on-write: before writing positions [start, end), unshare any block in that range."""
        table = self.tables[seq_id]
        for i in range(start // self.block_size, self.blocks_for(end)):
            bid = table[i]
            if self.ref[bid] > 1:
                new = self._take_free()
                if self.copy_fn is not None:
                    self.copy_fn(bid, new)
                self.ref[bid] -= 1
                table[i] = new

    def register_computed(self, seq_id: int, tokens: Sequence[int], num_computed: int, extra_keys=None) -> None:
        """Give every newly full block a hash, so later requests can find it."""
        if not self.enable_prefix_caching:
            return
        hashes = self.seq_hashes[seq_id]
        full = num_computed // self.block_size
        if full <= len(hashes):
            return
        new = block_hashes(tokens[: full * self.block_size], self.block_size, extra_keys, self.chained_hash)
        table = self.tables[seq_id]
        for i in range(len(hashes), full):
            h = new[i]
            if h not in self.hash_to_block and self.block_hash[table[i]] is None:
                self.hash_to_block[h] = table[i]
                self.block_hash[table[i]] = h
            hashes.append(h)

    def fork(self, parent_id: int, child_id: int) -> None:
        """Parallel sampling or beam search: the child shares every block, nothing is copied yet."""
        self.tables[child_id] = list(self.tables[parent_id])
        self.seq_hashes[child_id] = list(self.seq_hashes.get(parent_id, []))
        for bid in self.tables[child_id]:
            self.ref[bid] += 1

    def free_seq(self, seq_id: int) -> None:
        """Release in reverse order, so a sequence's last blocks are evicted before its prefix."""
        for bid in reversed(self.tables.pop(seq_id)):
            self._release(bid)
        self.seq_hashes.pop(seq_id, None)

    def check_invariants(self) -> None:
        counted = [0] * self.num_blocks
        for table in self.tables.values():
            for bid in table:
                counted[bid] += 1
        assert counted == self.ref, "reference counts do not match block tables"
        assert all((self.ref[b] == 0) == (b in self.free) for b in range(self.num_blocks))
        for h, bid in self.hash_to_block.items():
            assert self.block_hash[bid] == h
