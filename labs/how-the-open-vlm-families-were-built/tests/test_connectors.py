import torch

from vlmfam import connectors as C


def test_shuffles_match_hf_and_hold_the_same_values():
    from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector
    from transformers.models.internvl.modeling_internvl import InternVLModel

    x = torch.randn(2, 64, 6)
    a = C.pixel_shuffle(x, 2)
    assert torch.equal(a, Idefics3Connector.pixel_shuffle(None, x, 2))
    g = x.reshape(2, 8, 8, 6)
    b = C.internvl_pixel_shuffle(g)
    assert torch.equal(b, InternVLModel.pixel_shuffle(None, g, 0.5))
    assert a.shape == (2, 16, 24) and b.shape == (2, 4, 4, 24)
    assert torch.equal(a.sort(-1).values, b.reshape(2, 16, 24).sort(-1).values)


def test_patch_merger_matches_hf():
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger

    hf, ours = PatchMerger(dim=10, context_dim=6), C.PatchMerger(6, 10)
    ours.load_state_dict({"ln.weight": hf.ln_q.weight, "ln.bias": hf.ln_q.bias, "fc1.weight": hf.mlp[0].weight,
                          "fc1.bias": hf.mlp[0].bias, "fc2.weight": hf.mlp[2].weight, "fc2.bias": hf.mlp[2].bias})
    p = torch.randn(32, 6)
    assert torch.allclose(ours(p), hf(p), atol=1e-6)


def test_resampler_output_length_is_fixed():
    m = C.QueryResampler(12, 16, num_queries=5, heads=4)
    for n in (9, 100):
        assert m(torch.randn(2, n, 12)).shape == (2, 5, 16)


def test_formulas_match_module_parameter_counts():
    cfg = {"vision_config": {"embed_dim": 12, "hidden_size": 16, "spatial_merge_size": 2}}
    assert C.formula_params("qwen2_vl", cfg)[0] == C.n_params(C.PatchMerger(12, 16))
    cfg = {"vision_config": {"hidden_size": 12}, "text_config": {"hidden_size": 16}, "downsample_ratio": 0.5}
    assert C.formula_params("internvl", cfg)[0] == C.n_params(C.MLPConnector(48, 16, pre_norm=True))
    assert C.formula_params("llava", cfg)[0] == C.n_params(C.MLPConnector(12, 16))
    assert C.formula_params("gemma3", cfg)[0] == C.n_params(C.AvgPoolConnector(12, 16))
    cfg = {"vision_config": {"hidden_size": 12}, "hidden_size": 16, "query_num": 5}
    assert C.formula_params("minicpmv", cfg)[0] == C.n_params(C.QueryResampler(12, 16, num_queries=5, heads=4))
