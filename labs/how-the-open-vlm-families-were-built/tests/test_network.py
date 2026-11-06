"""Downloads processor files or reads safetensors headers from the Hub (no weights)."""

import pytest

pytestmark = pytest.mark.network


def test_qwen2_vl_processor_matches_rule():
    from vlmfam import processors as P

    probe = P.Probe("Qwen2/2.5-VL", "Qwen/Qwen2-VL-2B-Instruct")
    for w, h in [(1700, 2200), (300, 300), (4032, 3024)]:
        assert probe.measure(w, h) == probe.rule(w, h)


def test_smolvlm_256m_split_from_headers():
    from vlmfam import split as S

    parts = S.split(S.tensor_sizes("HuggingFaceTB/SmolVLM-256M-Instruct"))
    assert sum(parts.values()) == 256_484_928
    assert parts["connector"] == 768 * 16 * 576  # pixel shuffle x16, one bias-free linear
    assert parts["adapter"] == 0
