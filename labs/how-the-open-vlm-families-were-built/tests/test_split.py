from vlmfam import split as S


def test_classify_known_names():
    assert S.classify("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight") == "vision"
    assert S.classify("model.visual.blocks.3.attn.qkv.weight") == "vision"
    assert S.classify("visual.merger.mlp.0.weight") == "connector"
    assert S.classify("model.visual.deepstack_merger_list.0.norm.weight") == "connector"
    assert S.classify("multi_modal_projector.linear_1.weight") == "connector"
    assert S.classify("model.connector.modality_projection.proj.weight") == "connector"
    assert S.classify("resampler.attn.in_proj_weight") == "connector"
    assert S.classify("mlp1.1.weight") == "connector"
    assert S.classify("language_model.model.layers.0.self_attn.q_proj.weight") == "language"
    assert S.classify("lm_head.weight") == "language"


def test_cross_attention_layers_become_adapter():
    sizes = {
        "language_model.model.layers.0.self_attn.q_proj.weight": 10,
        "language_model.model.layers.3.cross_attn.q_proj.weight": 7,
        "language_model.model.layers.3.mlp.up_proj.weight": 5,
        "language_model.model.layers.30.mlp.up_proj.weight": 2,  # prefix 'layers.3' must not swallow 'layers.30'
        "vision_model.transformer.layers.0.mlp.fc1.weight": 4,
        "multi_modal_projector.weight": 1,
    }
    parts = S.split(sizes)
    assert parts == {"vision": 4, "connector": 1, "adapter": 12, "language": 12}
    assert abs(S.vision_share(parts) - 4 / 29) < 1e-12
