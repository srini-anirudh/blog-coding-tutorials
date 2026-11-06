import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("perception_probe", Path(__file__).resolve().parents[1] / "scripts" / "perception_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_probe_reads_counts_that_are_linear_in_the_features_and_have_a_nonzero_mean():
    """Counts are 0..3 with a mean near 1.3. Without an intercept the probe scored 0.000 on features that held the answer."""
    g = torch.Generator().manual_seed(0)
    counts = torch.randint(0, 4, (300, 3), generator=g).float()
    mix = torch.randn(3, 64, generator=g)
    feats = counts @ mix + 0.05 * torch.randn(300, 64, generator=g)
    assert probe.probe_accuracy(feats[:200], counts[:200], feats[200:], counts[200:]) > 0.9


def test_probe_is_at_chance_on_features_that_hold_nothing():
    g = torch.Generator().manual_seed(1)
    counts = torch.randint(0, 4, (300, 3), generator=g).float()
    noise = torch.randn(300, 64, generator=g)
    assert probe.probe_accuracy(noise[:200], counts[:200], noise[200:], counts[200:]) < 0.45
