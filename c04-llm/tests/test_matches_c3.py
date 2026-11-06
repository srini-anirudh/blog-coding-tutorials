"""tinyllm's attention is a trimmed copy of C3's attnzoo.GroupedQueryAttention. Check they agree.

Skipped when ../c03-attention is not present, so this folder still works on its own.
"""

import sys
from pathlib import Path

import pytest
import torch

from c04_helpers import small_config
from tinyllm import KVCache
from tinyllm.attention import GroupedQueryAttention
from tinyllm.rope import RotaryEmbedding

C3 = Path(__file__).resolve().parents[2] / "c03-attention"
pytestmark = pytest.mark.skipif(not (C3 / "attnzoo").exists(), reason="C3 attnzoo not available")


@pytest.fixture(scope="module")
def attnzoo():
    sys.path.insert(0, str(C3))
    import attnzoo

    return attnzoo


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_same_output_as_attnzoo(attnzoo, num_kv_heads):
    config = small_config(num_kv_heads=num_kv_heads)
    torch.manual_seed(0)
    ours = GroupedQueryAttention(config, layer_idx=0).eval()
    theirs = attnzoo.GroupedQueryAttention(
        attnzoo.AttentionConfig(
            hidden_size=config.hidden_size, num_heads=config.num_heads, num_kv_heads=num_kv_heads,
            max_seq_len=config.max_seq_len, rope_theta=config.rope_theta,
        )
    ).eval()
    theirs.load_state_dict(ours.state_dict())  # same parameter names: q_proj, k_proj, v_proj, o_proj

    x = torch.randn(2, 12, config.hidden_size)
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
    cos, sin = rope(torch.arange(12).expand(2, 12))
    torch.testing.assert_close(ours(x, cos, sin), theirs(x), atol=1e-5, rtol=1e-5)

    # Same cache protocol too: prefill 8 tokens, then decode 4 one at a time.
    our_cache, their_cache = KVCache(), attnzoo.KVCache()
    ours(x[:, :8], cos[:, :8], sin[:, :8], our_cache)
    theirs(x[:, :8], kv_cache=their_cache)
    for t in range(8, 12):
        a = ours(x[:, t : t + 1], cos[:, t : t + 1], sin[:, t : t + 1], our_cache)
        b = theirs(x[:, t : t + 1], kv_cache=their_cache)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert our_cache.get_seq_length() == their_cache.get_seq_length() == 12
