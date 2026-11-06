import pytest

from overparam import memorize


@pytest.mark.network
def test_load_cifar_subset():
    x, y, xt, yt = memorize.load_cifar(64, 32)
    assert x.shape == (64, 3072) and xt.shape == (32, 3072)
    assert abs(float(x.mean())) < 0.2 and set(y.tolist()) <= set(range(10))
