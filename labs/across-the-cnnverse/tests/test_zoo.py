import pytest
import torch

from cnnverse import zoo
from cnnverse.profile import profile, head_param_share, n_params, saved_activation_bytes


@pytest.mark.parametrize("name", list(zoo.NINE) + ["ResNeXt-50 32x4d", "SE-ResNet-50"])
def test_counts_within_published(name):
    build, inp, pub_p, pub_m, _ = {**zoo.NINE, **zoo.EXTRAS}[name]
    prof = profile(build(), inp)
    assert abs(prof.params / 1e6 / pub_p - 1) < 0.01
    assert abs(prof.macs / 1e9 / pub_m - 1) < (0.05 if name == "LeNet-5" else 0.025)


def test_exact_counts():
    assert n_params(zoo.lenet5()) == 61_706
    assert n_params(zoo.lenet5(paper=True)) == 60_840
    assert n_params(zoo.vgg16()) == 138_357_544
    assert n_params(zoo.resnet50()) == 25_557_032
    assert round(head_param_share(zoo.alexnet()), 2) == 0.96
    assert round(head_param_share(zoo.vgg16()), 2) == 0.89


def test_grouping_saves_alexnet_cost():
    g, u = profile(zoo.alexnet(), (3, 227, 227)), profile(zoo.alexnet(False), (3, 227, 227))
    assert round(u.params / 1e6, 1) == 62.4 and round(u.macs / 1e9, 2) == 1.14 and round(g.macs / 1e9, 2) == 0.72


def test_plain_and_residual_have_identical_weights():
    for d in (20, 56):
        assert n_params(zoo.cifar_net(d, True)) == n_params(zoo.cifar_net(d, False))
    assert round(n_params(zoo.cifar_net(20)) / 1e6, 2) == 0.27
    assert round(n_params(zoo.cifar_net(56)) / 1e6, 2) == 0.85


def test_real_forward_shapes():
    assert zoo.lenet5(paper=True)(torch.randn(2, 1, 32, 32)).shape == (2, 10)
    assert zoo.cifar_net(8)(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    for build in (zoo.mobilenet_v2, zoo.efficientnet, zoo.convnext_tiny, zoo.densenet121):
        assert build().eval()(torch.randn(1, 3, 64, 64)).shape == (1, 1000)


def test_saved_activations_scale_with_batch():
    m = zoo.cifar_net(8).train()
    one, four = saved_activation_bytes(m, torch.randn(1, 3, 32, 32)), saved_activation_bytes(m, torch.randn(4, 3, 32, 32))
    assert 3.5 < four / one < 4.1
