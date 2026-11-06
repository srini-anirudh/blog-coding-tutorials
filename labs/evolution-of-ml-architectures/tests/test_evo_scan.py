import math

import pytest
import torch

from archlab.scan import causal_conv, lti_kernel, parallel_scan, semiseparable_matrix, sequential_scan


@pytest.mark.parametrize("n", [1, 2, 7, 64, 100])
def test_parallel_scan_matches_sequential(n):
    a, b = torch.rand(3, n, 5), torch.randn(3, n, 5)
    h, rounds = parallel_scan(a, b)
    assert torch.allclose(h, sequential_scan(a, b), atol=1e-5)
    assert rounds == (math.ceil(math.log2(n)) if n > 1 else 0)


def test_lti_recurrence_is_a_convolution():
    n, d, s = 50, 3, 4
    a, B, C = torch.rand(d, s) * 0.5 + 0.4, torch.randn(d, s), torch.randn(d, s)
    x = torch.randn(2, n, d)
    y = (sequential_scan(a.expand(2, n, d, s), B * x[..., None]) * C).sum(-1)
    K = lti_kernel(a, B, C, n)
    assert torch.allclose(y, causal_conv(x, K), atol=1e-4)
    assert torch.allclose(y, causal_conv(x, K, fft=True), atol=1e-4)


def test_semiseparable_matrix_equals_scan():
    n, s = 30, 6
    a = torch.rand(n) * 0.5 + 0.5
    B, C, x = torch.randn(n, s), torch.randn(n, s), torch.randn(n)
    h = sequential_scan(a[None, :, None].expand(1, n, s), (B * x[:, None])[None])[0]
    assert torch.allclose((h * C).sum(-1), semiseparable_matrix(a, B, C) @ x, atol=1e-4)
