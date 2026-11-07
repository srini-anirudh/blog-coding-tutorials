import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from prefopt.goodhart import fit_form, fit_proxy, make_world, optimisation_curves, peak, sample_pairs

LAB = Path(__file__).resolve().parents[1]


def test_world_is_normalised_and_gold_peaks_at_length_one():
    w = make_world(n=500)
    assert torch.isclose(w.ref.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert w.gold.shape == (500,)


def test_overoptimisation_rises_then_falls():
    w = make_world(n=1500)
    win, los = sample_pairs(w, 800, seed=1)
    proxy = fit_proxy(w, win, los, width=16, steps=600)
    c = optimisation_curves(w, proxy)
    for key in ("bon", "kl"):
        p = peak(c[key])
        assert p["gold"] > 0  # some optimisation helps
        assert c[key][-1]["gold"] < p["gold"]  # too much hurts
        assert c[key][-1]["proxy"] > p["proxy"]  # while the proxy keeps rising
    a, b, _ = fit_form(c["bon"], "bon")
    assert a > 0 and b > 0  # the fitted curve also rises, then turns down


def test_fit_form_recovers_known_coefficients():
    pts = [{"d": d, "gold": d * (1.0 - 0.3 * d)} for d in torch.linspace(0.1, 3, 20).tolist()]
    a, b, r2 = fit_form(pts, "bon")
    assert abs(a - 1.0) < 1e-6 and abs(b - 0.3) < 1e-6 and r2 > 0.999999


@pytest.mark.slow
@pytest.mark.parametrize("script,args", [("01_bradley_terry.py", []), ("02_kl_tilt.py", []),
                                          ("06_loss_family.py", [])])
def test_scripts_run(script, args, tmp_path):
    out = subprocess.run([sys.executable, str(LAB / "scripts" / script), *args], capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr
    assert "# hardware:" in out.stdout


def test_saved_results_start_with_hardware_and_command():
    for f in sorted((LAB / "results").glob("*.txt")):
        head = f.read_text().splitlines()[:2]
        assert head[0].startswith("# hardware: ") and head[1].startswith("# command: python "), f.name
    for f in sorted((LAB / "results").glob("*.json")):
        d = json.loads(f.read_text())
        assert "hardware" in d and "command" in d, f.name


@pytest.mark.network
def test_smollm_tokenizes_a_pair():
    from prefopt.lm import INSTRUCT_MODEL, collate_pairs, load_tokenizer, tokenize_pair

    tok = load_tokenizer(INSTRUCT_MODEL)
    p = {"prompt": [{"role": "user", "content": "Hi?"}], "chosen": [{"role": "assistant", "content": "Hello."}],
         "rejected": [{"role": "assistant", "content": "Go away."}]}
    b = collate_pairs([tokenize_pair(tok, p)], pad_id=tok.pad_token_id)
    assert b["input_ids"].shape[0] == 2 and b["completion_mask"].sum() > 0
