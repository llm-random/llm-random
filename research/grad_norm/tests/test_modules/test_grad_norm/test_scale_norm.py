import math
from unittest.mock import patch

import torch

from research.grad_norm.modules.grad_norm.scale_norm import (
    GradientScaleNormLayer,
    scale_norm_grad,
)


def test_forward_id():
    x = torch.randn(3, 4, 5)
    layer = GradientScaleNormLayer()
    y = layer(x)
    assert torch.equal(x, y)


def test_grad_logging():
    layer = GradientScaleNormLayer()
    layer.update_cache_for_logging = layer.logging_cache.__setitem__

    x = torch.randn(3, 4, 5, requires_grad=True)
    grad = torch.rand_like(x)

    y = layer(x)
    y.backward(grad)
    logs = layer.log_heavy()
    assert torch.equal(logs["activation_norms/mean"], torch.mean(torch.norm(x, dim=-1)))
    assert torch.equal(logs["activation_norms/std"], torch.std(torch.norm(x, dim=-1)))
    assert torch.equal(logs["raw_grad_norms/mean"], torch.mean(torch.norm(grad, dim=-1)))
    assert torch.equal(logs["raw_grad_norms/std"], torch.std(torch.norm(grad, dim=-1)))


def test_norm_grad_logging():
    k = 0.5
    eps = 1e-5
    layer = GradientScaleNormLayer(k=k, eps=eps)
    layer.update_cache_for_logging = layer.logging_cache.__setitem__

    x = torch.randn(3, 4, 5, requires_grad=True)
    grad = torch.rand_like(x)

    y = layer(x)
    y.backward(grad)

    logs = layer.log_heavy()
    norm_grad = torch.norm(scale_norm_grad(grad, k, eps), dim=-1)
    assert torch.equal(logs["norm_grad_norms/mean"], torch.mean(norm_grad))
    assert torch.equal(logs["norm_grad_norms/std"], torch.std(norm_grad))


@patch("research.grad_norm.modules.grad_norm.scale_norm.scale_norm_grad", wraps=scale_norm_grad)
def test_compute_k(scale_norm_grad_mock):
    layer = GradientScaleNormLayer(k="auto")
    x = torch.rand(3, 4, 5, requires_grad=True)
    grad = torch.rand_like(x)
    y = layer(x)
    y.backward(grad)

    expected_k = math.sqrt(x.shape[1] * x.shape[2])
    assert scale_norm_grad_mock.call_args[1]["k"] == expected_k


def test_backward_norm():
    layer = GradientScaleNormLayer()
    x = torch.randn(3, 4, 5, requires_grad=True)
    grad = torch.rand_like(x)
    y = layer(x)
    y.backward(grad)

    expected_grad = scale_norm_grad(grad, k=layer.k, eps=layer.eps)

    assert torch.allclose(x.grad, expected_grad)
