import torch

from servekit.alloc_sim import FirstFit, Req, sharing_blocks, simulate
from servekit.block_manager import BlockManager, block_hashes
from servekit.spec import acceptance_rate, expected_tokens, speculative_step


def test_lazy_allocation_and_bounded_waste():
    bm = BlockManager(100, block_size=16)
    bm.allocate_prompt(0, list(range(33)))
    assert bm.tables[0] == []  # nothing allocated at admission
    assert bm.ensure(0, 33)
    assert len(bm.tables[0]) == 3  # 33 tokens -> 3 blocks, at most 15 slots unused
    assert not bm.ensure(0, 16 * 101)  # all or nothing
    assert len(bm.tables[0]) == 3
    bm.free_seq(0)
    assert bm.num_free == 100
    bm.check_invariants()


def test_fork_shares_then_copy_on_write():
    copies = []
    bm = BlockManager(20, block_size=4)
    bm.copy_fn = lambda s, d: copies.append((s, d))
    bm.allocate_prompt(0, list(range(10)))
    bm.ensure(0, 10)  # blocks: [full, full, half]
    bm.fork(0, 1)
    assert bm.tables[0] == bm.tables[1] and bm.num_free == 17
    bm.ensure(1, 11)
    bm.prepare_write(1, 10, 11)  # the child writes into the shared half block
    assert len(copies) == 1 and copies[0][0] == bm.tables[0][2]
    assert bm.tables[0][:2] == bm.tables[1][:2] and bm.tables[0][2] != bm.tables[1][2]
    bm.prepare_write(0, 10, 11)  # the parent is now the only owner: no copy
    assert len(copies) == 1
    bm.check_invariants()


def test_chained_hash_prevents_cross_conversation_collision():
    shared_block = [7] * 4
    a = [1, 2, 3, 4] + shared_block
    b = [9, 9, 9, 9] + shared_block  # same block 1 content, different history
    assert block_hashes(a, 4)[1] != block_hashes(b, 4)[1]
    assert block_hashes(a, 4, chained=False)[1] == block_hashes(b, 4, chained=False)[1]


def test_unchained_cache_serves_wrong_kv():
    """With per-block hashes only, request b reuses a block whose KV was computed after a's history."""
    for chained, expect_hit in ((True, 0), (False, 8)):
        bm = BlockManager(20, 4, enable_prefix_caching=True, chained_hash=chained)
        a = [1, 2, 3, 4, 7, 7, 7, 7, 5]
        bm.allocate_prompt(0, a)
        bm.ensure(0, len(a))
        bm.register_computed(0, a, len(a))
        b = [7, 7, 7, 7, 7, 7, 7, 7, 5]  # both blocks of b equal block 1 of a, so both 'hit'
        assert bm.allocate_prompt(1, b) == expect_hit


def test_image_hash_in_block_key():
    """Placeholder ids are the same for every image. Without the image hash, two images collide."""
    tokens = [42] * 8
    assert block_hashes(tokens, 4, ["cat", "cat"]) != block_hashes(tokens, 4, ["dog", "dog"])
    assert block_hashes(tokens, 4) == block_hashes(tokens, 4)


def test_lazy_eviction_resurrects_freed_prefix():
    bm = BlockManager(4, 4, enable_prefix_caching=True)
    p = list(range(9))
    bm.allocate_prompt(0, p)
    bm.ensure(0, 9)
    bm.register_computed(0, p, 9)
    bm.free_seq(0)
    assert bm.num_free == 4 and len(bm.hash_to_block) == 2  # freed, but still findable
    assert bm.allocate_prompt(1, p) == 8
    bm.check_invariants()
    bm.free_seq(1)
    bm.allocate_prompt(2, [99] * 16)
    assert bm.ensure(2, 16)  # needs all 4 blocks: the cached ones are evicted now, not before
    assert len(bm.hash_to_block) == 0


def test_first_fit_fragments():
    ff = FirstFit(10)
    assert ff.alloc(0, 3) and ff.alloc(1, 3) and ff.alloc(2, 3)
    ff.free(0)
    ff.free(2)
    assert ff.largest_hole() == 4  # 7 free tokens, but no hole of 5
    assert not ff.alloc(3, 5)


def test_alloc_sim_orders_policies():
    reqs = [Req(50 + (i * 37) % 200, 10 + (i * 53) % 300, 400) for i in range(200)]
    res = {p: simulate(reqs, 8192, p, max_model_len=1024) for p in ("reserve_max", "contiguous", "paged")}
    assert res["paged"]["mean_concurrency"] > res["contiguous"]["mean_concurrency"] > res["reserve_max"]["mean_concurrency"]
    assert res["paged"]["kv_utilization"] > 0.9
    assert res["paged"]["steps"] < res["reserve_max"]["steps"]


def test_sharing_matches_post_arithmetic():
    r = sharing_blocks(prompt=2000, n=8, generated=1)
    assert r["prompt_blocks"] == 125
    assert r["shared_blocks"] == 125 + 8  # 2000 = 125 full shared blocks; each sample opens one private block
    r = sharing_blocks(prompt=2001, n=8, generated=1)
    assert r["shared_blocks"] == 125 + 8  # 125 shared full blocks plus 8 private partial blocks


def test_expected_tokens_formula():
    assert abs(expected_tokens(0.8, 4) - 3.3616) < 1e-4  # the post rounds this to 3.36


def test_speculative_sampling_is_exact():
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(3, 6, generator=g) * 2, -1)
    q = torch.softmax(torch.randn(2, 6, generator=g) * 2, -1)
    counts = torch.zeros(6)
    for _ in range(20000):
        counts[speculative_step(p, q, g)[0]] += 1
    assert (counts / counts.sum() - p[0]).abs().max() < 0.015
    assert 0 < acceptance_rate(p[0], q[0]) < 1
