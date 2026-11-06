import pytest

from pixelcost import costmodel as cm


def test_worked_example_matches_post():
    c = cm.request_cost(1008, 1344)
    assert (c.patches, c.visual, c.prompt) == (6912, 1728, 1768)
    assert c.encoder_flops / 1e12 == pytest.approx(9.3, abs=0.05)
    assert c.encoder_s * 1e3 == pytest.approx(19, abs=0.5)
    assert c.prefill_flops / 1e12 == pytest.approx(26, abs=0.1)
    assert c.prefill_s * 1e3 == pytest.approx(53, abs=0.5)
    assert cm.POST_LLM.kv_bytes_per_token() == 56 * 1024
    assert c.kv_bytes / 1e6 == pytest.approx(101, abs=0.5)
    assert round(c.image_kv_share, 2) == 0.98
    assert c.decode_step_s * 1e3 == pytest.approx(4.2, abs=0.05)


def test_no_gqa_is_seven_times_the_cache():
    mha = cm.LanguageModel(7e9, 28, 3584, 28, 128)
    assert mha.kv_bytes_per_token() * 1768 / 1e6 == pytest.approx(710, abs=1)
    assert mha.kv_bytes_per_token() / cm.POST_LLM.kv_bytes_per_token() == 7


def test_token_rules():
    assert cm.visual_tokens(336, 336, 14, 1) == 576          # LLaVA-1.5
    assert cm.visual_tokens(448, 448, 14, 2) == 256          # InternVL tile after pixel shuffle 2
    assert cm.anthropic_tokens(1000, 1000) == 1296
    assert cm.openai_tokens(512, 512) == 170 * 1 + 85
    assert cm.openai_tokens(1024, 1024) == 170 * 4 + 85      # shortest side scaled to 768: 2 x 2 tiles
    assert cm.gemini_tokens(300, 300) == 258


def test_pruning_and_answer_length_move_the_bottleneck():
    full, pruned = cm.request_cost(1008, 1344), cm.request_cost(1008, 1344, keep=0.25)
    assert pruned.visual == 432 and pruned.prefill_s < full.prefill_s / 3
    assert (full.total_s(1) - pruned.total_s(1)) / full.total_s(1) > 0.4                 # short answer: big win
    assert (full.total_s(200) - pruned.total_s(200)) / full.total_s(200) < 0.06            # long answer: small
    assert cm.concurrent_requests(pruned) > 2 * cm.concurrent_requests(full)
