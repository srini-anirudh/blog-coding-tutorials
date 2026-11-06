import pytest
import torch

from pixelcost import costmodel as cm
from pixelcost import qwen, selection, serving, synth, testing


@pytest.fixture(scope="module")
def tiny():
    model, inputs = testing.tiny_qwen2vl(), testing.tiny_inputs()
    p = qwen.prepare(model, inputs, qwen.encode(model, inputs))
    return model, inputs, p


def test_split_pipeline_matches_the_model_forward(tiny):
    model, inputs, p = tiny
    with torch.inference_mode():
        ref = model(**inputs).logits[:, -1]
    logits, _ = qwen.prefill(model, *qwen.reduce(p))
    assert torch.allclose(logits, ref, atol=1e-5)


def test_manual_greedy_decode_matches_generate(tiny):
    model, inputs, p = tiny
    with torch.inference_mode():
        ref = model.generate(**inputs, max_new_tokens=6, do_sample=False)[0, inputs["input_ids"].shape[1]:]
    e, pos = qwen.reduce(p)
    logits, cache = qwen.prefill(model, e, pos)
    assert qwen.decode(model, logits, cache, int(pos.max()) + 1, 6) == ref.tolist()


def test_attention_score_equals_eager_attention_weights(tiny):
    model, inputs, p = tiny
    e, pos = qwen.reduce(p)
    ours = qwen.attention_received(model, e, pos, layer=2)
    cfg = model.model.language_model.config
    cfg._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            out = model.model.language_model(inputs_embeds=e, position_ids=pos, output_attentions=True)
    finally:
        cfg._attn_implementation = "sdpa"
    ref = out.attentions[2][0, :, -1].mean(0)
    assert torch.allclose(ours, ref, atol=1e-5) and abs(float(ours.sum()) - 1) < 1e-5


def test_reduction_keeps_original_3d_positions(tiny):
    model, inputs, p = tiny
    red = selection.uniform_grid(p.embeds[0, p.image_slice], *p.grid, 6)
    e, pos = qwen.apply_reduction(p, red)
    n_removed = (p.grid[0] * p.grid[1]) - 6
    assert e.shape[1] == p.embeds.shape[1] - n_removed
    s = p.image_slice
    assert torch.equal(pos[:, :, s.start:s.start + 6], p.positions[:, :, s][:, :, red.index])
    assert torch.equal(pos[:, :, s.start + 6:], p.positions[:, :, s.stop:])     # text after keeps its positions


def test_placeholder_only_keys_collide_and_image_hash_separates():
    a, b = synth.solid_shape("red"), synth.solid_shape("blue", circle=False)
    ids = [1] * 5 + [90] * 40 + [2] * 10
    ha, hb = serving.image_hash(a), serving.image_hash(b)
    assert ha != hb and serving.image_hash(a, "max_pixels=1M") != ha
    assert serving.block_hashes(ids, 16) == serving.block_hashes(ids, 16)
    assert serving.block_hashes(ids, 16, 90, ha) != serving.block_hashes(ids, 16, 90, hb)
    assert serving.block_hashes([1] * 16 + ids, 16, 90, ha)[0] == serving.block_hashes([1] * 16 + ids, 16, 90, hb)[0]


def test_prefix_cache_reuse_gives_the_same_logits(tiny):
    model, inputs, p = tiny
    e, pos = qwen.reduce(p)
    ids = inputs["input_ids"][0].tolist()
    full, kv = qwen.prefill(model, e, pos)
    cache = serving.PrefixCache(block=8)
    cache.insert(ids, testing.IMAGE_TOKEN, "img", kv)
    matched, kv2 = cache.lookup(ids, testing.IMAGE_TOKEN, "img")
    assert matched == len(ids) // 8 * 8 and kv2.get_seq_length() == matched
    logits, _ = qwen.prefill(model, e[:, matched:], pos[:, :, matched:], cache=kv2)
    assert torch.allclose(logits, full, atol=1e-5)
    assert cache.lookup(ids, testing.IMAGE_TOKEN, "other image")[0] == 0


def test_synthetic_samples_are_deterministic():
    s1, s2 = synth.make_sample(3), synth.make_sample(3)
    assert (s1.color, s1.number, s1.band) == (s2.color, s2.number, s2.band)
    assert synth.is_correct("number", s1, f" {s1.number[:2]} {s1.number[2:]}")
    assert not synth.is_correct("color", s1, "none")


@pytest.mark.network
@pytest.mark.parametrize("hw", [(448, 448), (672, 672), (1008, 1344)])
def test_qwen2vl_processor_token_count_matches_cost_model(hw):
    processor = qwen.load_processor()
    img = synth.make_sample(0, side=max(hw)).image.resize((hw[1], hw[0]))
    inputs = qwen.build_inputs(processor, img, "hi")
    assert int(inputs["image_grid_thw"][0].prod()) == cm.num_patches(*hw)
    assert int(inputs["mm_token_type_ids"].sum()) == cm.visual_tokens(*hw)
