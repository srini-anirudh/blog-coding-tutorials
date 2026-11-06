import pytest
import torch

from cnnverse import data, zoo


@pytest.mark.network
def test_cifar10_loads():
    d = data.load_cifar10()
    assert d["train"][0].shape == (50000, 3, 32, 32) and d["test"][0].shape == (10000, 3, 32, 32)
    assert d["train"][0].dtype == torch.uint8 and int(d["train"][1].max()) == 9


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    torch.manual_seed(0)
    m, x = zoo.cifar_net(8).eval(), torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)
        torch.backends.cudnn.allow_tf32 = False
        got = m.cuda()(x.cuda()).cpu()
    assert torch.allclose(ref, got, atol=1e-4)
