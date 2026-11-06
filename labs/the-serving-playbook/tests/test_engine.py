"""The engine must produce exactly the tokens of plain greedy decoding, whatever the scheduler does."""

import pytest
import torch

from servekit.engine import Engine
from servekit.metrics import random_prompts
from servekit.model import load_model
from servekit.scheduler import SchedulerConfig


@pytest.fixture(scope="module")
def model():
    return load_model("tiny")


def reference(model, prompt, n):
    """No cache at all: rerun the full sequence through C4's TinyLLM for every token."""
    ids = torch.tensor([prompt])
    for _ in range(n):
        nxt = model.base(ids).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids[0, len(prompt):].tolist()


def run(model, prompts, lens, **kw):
    sched = kw.pop("sched", SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64))
    eng = Engine(model, num_blocks=kw.pop("num_blocks", 256), block_size=kw.pop("block_size", 8), sched=sched, **kw)
    seqs = [eng.add_request(p, n) for p, n in zip(prompts, lens)]
    while eng.sched.has_work():
        eng.step()
    eng.bm.check_invariants()
    assert eng.bm.num_free == eng.bm.num_blocks
    return eng, [s.output[: s.max_new_tokens] for s in seqs]


PROMPTS = random_prompts(6, 3, 40, 256, seed=1)
LENS = [5, 12, 3, 9, 7, 4]


@pytest.fixture(scope="module")
def expected(model):
    return [reference(model, p, n) for p, n in zip(PROMPTS, LENS)]


def test_continuous_chunked_matches_reference(model, expected):
    _, out = run(model, PROMPTS, LENS)
    assert out == expected


def test_unchunked_and_static_match_reference(model, expected):
    for cfg in (
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512, chunked_prefill=False),
        SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=512, chunked_prefill=False, mode="static"),
    ):
        _, out = run(model, PROMPTS, LENS, sched=cfg)
        assert out == expected


def test_tiny_chunks_still_exact(model, expected):
    """A 5-token budget forces prompts across many iterations, mixed with other sequences' decodes."""
    _, out = run(model, PROMPTS, LENS, sched=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=5))
    assert out == expected


def test_preemption_by_recompute_is_exact(model, expected):
    eng, out = run(model, PROMPTS, LENS, num_blocks=12, watermark=0.0)
    assert eng.sched.num_preemptions > 0
    assert out == expected


def test_prefix_cache_hits_and_stays_exact(model):
    system = random_prompts(1, 40, 40, 256, seed=7)[0]
    prompts = [system + q for q in random_prompts(5, 2, 6, 256, seed=8)]
    want = [reference(model, p, 4) for p in prompts]
    # One at a time, so later requests find the system prompt already cached.
    eng, out = run(model, prompts, [4] * 5, enable_prefix_caching=True,
                   sched=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64))
    assert out == want
    assert eng.sched.cached_tokens == 4 * 40  # 40 = five full blocks of 8, found by requests 2 to 5


def test_static_mode_squats(model):
    eng, _ = run(model, PROMPTS[:3], [2, 2, 10],
                 sched=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=512, chunked_prefill=False, mode="static"))
    decode_steps = [s for s in eng.stats if s.num_decodes]
    assert all(s.num_decodes == 3 for s in decode_steps)  # finished members keep their slots
    assert decode_steps[-1].num_useful_decodes == 1
