import pytest
import torch

from gallery.cost import kv_bytes_per_token, measure_kv_bytes, picture_cost


def test_post_numbers():
    assert kv_bytes_per_token(32, 8, 128) == 128 * 1024
    got = {t: picture_cost(t) for t in (576, 2880, 1300, 10240)}
    assert [got[t]["kv_mib"] for t in (576, 2880, 1300, 10240)] == [72.0, 360.0, 162.5, 1280.0]
    assert [got[t]["images_in_context"] for t in (576, 2880, 1300, 10240)] == [222, 44, 98, 12]


@pytest.mark.parametrize("layers,kv,hd,tok", [(2, 2, 16, 10), (4, 1, 32, 33)])
def test_formula_matches_real_cache(layers, kv, hd, tok):
    assert measure_kv_bytes(tok, layers, kv, hd, dtype=torch.bfloat16) == tok * kv_bytes_per_token(layers, kv, hd)
    assert measure_kv_bytes(tok, layers, kv, hd, dtype=torch.float32) == tok * kv_bytes_per_token(layers, kv, hd, 4)


@pytest.mark.network
def test_hub_configs_agree_with_post():
    from gallery.hub import check_all

    rows = {r["model"]: r for r in check_all()}
    for name, r in rows.items():
        assert "error" not in r, (name, r.get("error"))
    assert rows["LLaVA-1.5"]["tokens_from_config"] == 576
    assert rows["IDEFICS3"]["tokens_from_config"] == 169
    assert rows["InternVL 2.5"]["tokens_from_config"] == 256
    assert rows["Qwen2-VL"]["tokens_from_config"] == 16384
    # the post's "up to 2 880" leaves out the unpadding and the newline tokens HF adds
    assert rows["LLaVA-NeXT"]["tokens_from_config"] == 2928
