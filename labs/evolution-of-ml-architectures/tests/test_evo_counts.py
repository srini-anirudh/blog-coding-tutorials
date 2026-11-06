import torch
from transformers import LlamaConfig, LlamaForCausalLM, MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer

from archlab import counts as C


def test_family_formulas_match_modules():
    d = 64
    f, m = C.family_formulas(d), C.family_modules(d)
    for name in ["MLP", "Conv 3x3", "Depthwise sep. 3x3", "Elman RNN", "Transformer"]:
        assert f[name] == C.n_params(m[name]), name
    # torch keeps b_ih and b_hh: one extra bias vector per gate
    assert C.n_params(m["GRU"]) == f["GRU"] + 3 * d
    assert C.n_params(m["LSTM"]) == f["LSTM"] + 4 * d


def test_mamba_formula_matches_hf_mixer():
    d = 128
    with torch.device("meta"):
        mixer = MambaMixer(MambaConfig(hidden_size=d, state_size=16, expand=2, conv_kernel=4, time_step_rank=8,
                                       use_bias=False, use_conv_bias=True), layer_idx=0)
    assert C.n_params(mixer) == C.mamba_block_params(d, dt_rank=8)


def test_llama3_8b_from_six_numbers():
    cfg = C.LLAMA3_8B
    total = C.decoder_params(**cfg)["total"]
    assert total == 8_030_261_248
    with torch.device("meta"):
        model = LlamaForCausalLM(LlamaConfig(vocab_size=cfg["V"], hidden_size=cfg["d"], intermediate_size=cfg["d_ff"],
                                             num_hidden_layers=cfg["L"], num_attention_heads=cfg["n_h"],
                                             num_key_value_heads=cfg["n_kv"], tie_word_embeddings=False))
    assert C.n_params(model) == total


def test_bytes_and_flops():
    assert C.kv_cache_bytes(32, 1, 8, 128) == 128 * 1024
    assert C.training_bytes(10) == 160
    assert abs(C.attention_to_param_flops(6 * 4096, 4096) - 1.0) < 1e-12


def test_cnn_history_split():
    conv, dense = C.split_conv_dense(C.alexnet())
    assert 3.7e6 < conv < 3.8e6 and 58.6e6 < dense < 58.7e6
    conv, dense = C.split_conv_dense(C.vgg16())
    assert conv + dense == 138_357_544
    assert C.n_params(C.lenet5()) == 61_706
