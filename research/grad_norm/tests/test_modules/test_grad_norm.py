from functools import partial
from typing import Any, Callable, Union
from unittest.mock import Mock

import pytest
import torch

from lizrd.core.misc import LoggingLayer
from research.grad_norm.modules.grad_norm import (
    BaseGradientSTDNormFunction,
    GradientSTDNormLayerV1,
    GradientSTDNormLayerV2,
    GradientSTDNormLayerV3,
    std_grad_norm_v1,
    std_grad_norm_v2,
    std_grad_norm_v3,
)


def get_std_grad_norm_fn_from_layer(
    layer: Union[GradientSTDNormLayerV1, GradientSTDNormLayerV2, GradientSTDNormLayerV3]
) -> Callable[[torch.Tensor, Any, float, float], torch.Tensor]:
    if isinstance(layer, GradientSTDNormLayerV3):
        return std_grad_norm_v3
    elif isinstance(layer, GradientSTDNormLayerV2):
        return std_grad_norm_v2
    else:
        return std_grad_norm_v1


@pytest.mark.parametrize(
    "grad_norm_layer", [GradientSTDNormLayerV1(c=1), GradientSTDNormLayerV2(c=1), GradientSTDNormLayerV3(c=1)]
)
class TestGradNormLayerCommonProperties:
    def test_forward_id(self, grad_norm_layer: LoggingLayer):

        x = torch.randn(3, 4, 4)
        y = grad_norm_layer(x)

        assert x.shape == y.shape
        assert torch.equal(x, y)

    def test_backward_unary(self, grad_norm_layer: LoggingLayer):

        x = torch.randn(1, requires_grad=True)
        grad = torch.randn(1)
        y = grad_norm_layer(x)
        y.backward(grad)

        assert torch.equal(x.grad, grad)

    def test_grad_logging(self, grad_norm_layer: LoggingLayer):
        grad_norm_layer.update_cache_for_logging = grad_norm_layer.logging_cache.__setitem__

        x = torch.randn(3, 4, 4, requires_grad=True)
        grad = torch.rand_like(x)
        y = grad_norm_layer(x)
        y.backward(grad)

        logs = grad_norm_layer.log_heavy()
        assert torch.equal(logs["activation_norms/mean"], torch.mean(torch.norm(x, dim=-1)))
        assert torch.equal(logs["activation_norms/std"], torch.std(torch.norm(x, dim=-1)))
        assert torch.equal(logs["raw_grad_norms/mean"], torch.mean(torch.norm(grad, dim=-1)))
        assert torch.equal(logs["raw_grad_norms/std"], torch.std(torch.norm(grad, dim=-1)))

    @pytest.mark.skip("not sure if this should be tested")
    @pytest.mark.parametrize("c", [i / 10 for i in range(1, 10)])  # from 0.1 to 0.9
    def test_grad_reduces_std(self, grad_norm_layer: LoggingLayer, c: float):
        grad_norm_layer.update_cache_for_logging = grad_norm_layer.logging_cache.__setitem__
        grad_norm_layer.c = c

        grad = torch.randn(10, 10, 10, requires_grad=True)
        y = grad_norm_layer(grad)
        y.backward(grad)

        logs = grad_norm_layer.log_heavy()
        assert abs(logs["norm_grad_norms/std"]) < abs(logs["raw_grad_norms/std"])

    def test_norm_grad_logging(self, grad_norm_layer: LoggingLayer):
        grad_norm_layer.update_cache_for_logging = grad_norm_layer.logging_cache.__setitem__

        x = torch.randn(3, 4, 4, requires_grad=True)
        grad = torch.rand_like(x)
        y = grad_norm_layer(x)
        y.backward(grad)

        logs = grad_norm_layer.log_heavy()
        std_grad_norm_fn = get_std_grad_norm_fn_from_layer(grad_norm_layer)
        norm_grad = torch.norm(std_grad_norm_fn(grad, c=grad_norm_layer.c, eps=grad_norm_layer.eps), dim=-1)
        assert torch.equal(logs["norm_grad_norms/mean"], torch.mean(norm_grad))
        assert torch.equal(logs["norm_grad_norms/std"], torch.std(norm_grad))

    def test_backward_norm(self, grad_norm_layer: LoggingLayer):
        x = torch.randn(3, 4, requires_grad=True)
        grad = torch.randn(3, 4)
        y = grad_norm_layer(x)
        y.backward(grad)

        std_grad_norm_fn = get_std_grad_norm_fn_from_layer(grad_norm_layer)
        with torch.no_grad():
            expected_grad = std_grad_norm_fn(grad, c=grad_norm_layer.c, eps=grad_norm_layer.eps)

        assert torch.allclose(x.grad, expected_grad)

    @pytest.mark.skip("not sure if in this case gradcheck is working")
    def test_pass_gradcheck(self, grad_norm_layer: LoggingLayer):
        c = 1
        eps = 1e-8
        x = torch.randn(3, 4, 4, requires_grad=True, dtype=torch.float64)
        std_grad_norm_fn = get_std_grad_norm_fn_from_layer(grad_norm_layer)
        assert torch.autograd.gradcheck(
            BaseGradientSTDNormFunction.apply,
            (x, Mock(), partial(std_grad_norm_fn, c=c, eps=eps)),
            raise_exception=True,
        )
