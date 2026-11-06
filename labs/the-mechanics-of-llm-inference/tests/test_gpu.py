"""CUDA-only checks: the CUDA-graph replay used by every benchmark gives the eager step's logits."""

import pytest
import torch

from inferlab.decoder import SHAPES, Decoder
from inferlab.timing import cuda_graph

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replay_matches_eager():
    m = Decoder.random(SHAPES["tiny"], 64, device="cuda", dtype=torch.bfloat16)
    m.allocate_cache(2)
    ids = torch.randint(0, 256, (2, 33), device="cuda")
    m.forward(ids[:, :32], 0)
    eager = m.forward(ids[:, 32:], 32).clone()
    out = {}

    def step():
        out["y"] = m.forward(ids[:, 32:], 32)

    cuda_graph(step)()
    torch.testing.assert_close(out["y"], eager)
